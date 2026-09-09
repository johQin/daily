# Ansible

# 0 绪论

## 0.1 初体验

```bash
# 在主控电脑是上生成ssh密钥对，本次生成的密钥对的名称是id_ansi.pub，id_ansi，如果没有生成在~/.ssh下，请手动copy到其中
ssh-keygen -t ed25519

# 将公钥id_ansi.pub，copy到被控主机上，user是被控主机账号名，your_server_ip是被控主机ip，在命令执行时，会让输入被控主机的密码
ssh-copy-id -i ~/.ssh/id_ansi user@your_server_ip
# 在命令执行成功后，会在被控主机创建或编辑 ~/.ssh/authorized_keys 文件，并将你的公钥内容追加到该文件末尾（不会覆盖已有的其他公钥）。
# ssh user@your_server_ip 来验证是否需要密码？
# 1. 如果依然问密码
# 	a. 可能是被控服务器的 /etc/ssh/sshd_config 中禁用了公钥认证（需检查 PubkeyAuthentication yes 是否开启），在开启之后，重启sudo systemctl restart sshd服务
# 	b. 查看 ~/.ssh/authorized_keys，确认你的本地公钥确实在里面
# 	c. SSH客户端不会主动识别一个名叫id_ansi的密钥对，所以你在ssh -i ~/.ssh/id_ansi ubuntu@10.0.0.8 就直接不需要密码，而在ssh ubuntu@10.0.0.8就需要密码，详细如下
#SSH 客户端（你的本地电脑）在连接时，默认只会自动尝试以下几把“标准名字”的私钥：
# ~/.ssh/id_rsa，~/.ssh/id_ecdsa，~/.ssh/id_ed25519，~/.ssh/id_dsa
#而你生成的那把密钥叫 id_ansi，这不在 SSH 的“默认白名单”里。因此，当你执行 ssh ubuntu@10.0.0.8 时：
# SSH 客户端尝试发送默认的 ~/.ssh/id_rsa（如果存在的话）。
# 服务器的 authorized_keys 里没有 id_rsa.pub 的内容，拒绝。
#SSH 客户端发现默认密钥失败，又没有其他指示，于是直接放弃公钥验证，转而询问服务器密码。
# 它从头到尾都没想起要把 id_ansi 发过去（除非你用 -i 强行指定）
# ssh -i ~/.ssh/id_ansi ubuntu@10.0.0.8 这样的时候，不需要密码

# 现在请在 ~/.ssh/config中配置，使ssh知道在连接被控主机时，主动携带对应的公钥
Host 10.0.0.8 10.0.0.9 10.0.0.10
    IdentityFile ~/.ssh/id_ansi
    User ubuntu
    
# 配置好~/.ssh/config后，在本地用 -G 参数（试运行）查看 SSH 最终读取的配置，看看是否生效
ssh -G 10.0.0.8 | grep -E "user|identityfile"
user ubuntu
identityfile ~/.ssh/id_ansi
userknownhostsfile /home/ubuntu/.ssh/known_hosts /home/ubuntu/.ssh/known_hosts2

# 安装ansible
sudo apt update
sudo apt install ansible -y

# 创建ansible 的主机清单inventory.ini，你可以按功能分组，便于后续批量操作
# Ansible 的 INI 格式清单支持 # 和 ; 两种注释符号。只要它们出现在行首，该行就会被完全忽略。
# 示例1：
[webservers]
web1 ansible_host=你的服务器1_IP ansible_user=root
web2 ansible_host=你的服务器2_IP ansible_user=root

[databases]
db1 ansible_host=你的服务器3_IP ansible_user=root
db2 ansible_host=你的服务器4_IP ansible_user=root

[all:children]
webservers
databases

# 示例2：
# 用于指定密钥，端口，用户名
[servers]
srv1 192.168.1.10 ansible_user=ubuntu ansible_port=22 ansible_ssh_private_key_file=~/.ssh/id_ed25519

# 示例3：
# 如果你三台机器的用户名都一样，且不想在 config 里写死，也可以在这里统一写一次
[webservers]
web1 ansible_host=10.0.0.8
web2 ansible_host=10.0.0.9
web3 ansible_host=10.0.0.10

[webservers:vars]
ansible_user=ubuntu
# 如果在~/.ssh/config中已配置，那么就无需写ansible_ssh_private_key_file


# 测试inventory.ini
ansible all -i inventory.ini -m ping
```

## 0.2 inventory.ini

它的语法规范部分：

- `[group_name]`: 中括号用于定义主机组的组名
  - 组名建议只使用 **字母、数字、下划线(`_`)**，且应以字母或下划线开头。
- `[group_name:特殊修饰符]`：特殊修饰符包含：vars，children
  - `[workers:vars]` 就是用来定义该组所有主机共享的变量。变量以 `key=value` 格式定义。
  - **`:children` - 定义子组**：用于将多个组组合成一个更大的父组。父组会自动包含所有子组中的主机
- **组嵌套**：组可以嵌套，但**不能形成循环引用**。
- **默认组**：`all` 和 `ungrouped` 是两个隐式存在的默认组

```ini
[all:children]
masters
workers

[all:vars]
ansible_user=ubuntu
ansible_port=20022

[masters]
master02 ansible_host=10.0.0.8
 
[workers] 
worker01 ansible_host=10.0.0.6
worker02 ansible_host=10.0.0.17

```

```bash
ansible all -i inventory.ini -m ping

worker02 | SUCCESS => {
    "ansible_facts": {
        "discovered_interpreter_python": "/usr/bin/python3"
    },
    "changed": false,
    "ping": "pong"
}
worker01 | SUCCESS => {
    "ansible_facts": {
        "discovered_interpreter_python": "/usr/bin/python3"
    },
    "changed": false,
    "ping": "pong"
}
master02 | SUCCESS => {
    "ansible_facts": {
        "discovered_interpreter_python": "/usr/bin/python3"
    },
    "changed": false,
    "ping": "pong"
}
```

## 0.3 ansilbe 默认配置优先级

Ansible 查找配置文件的优先级是固定的（从高到低）：

1. **环境变量**：`ANSIBLE_CONFIG` 指定的路径，`export ANSIBLE_CONFIG="~/ansi/ansible.cfg"`
2. **当前目录**：`./ansible.cfg`
3. **用户家目录**：`~/.ansible.cfg`
4. **全局目录**：`/etc/ansible/ansible.cfg`

ansible.cfg 示例：

```bash
[defaults]
# ========== 基础设置 ==========
# 指定主机清单文件的位置（可以是文件或目录）[reference:7][reference:8]
inventory = ./inventory

# 默认的远程连接用户[reference:9][reference:10]
remote_user = your_username

# 并发执行任务时的进程数，默认5[reference:11][reference:12]
forks = 10

# SSH连接超时时间（秒），默认10秒[reference:13][reference:14]
timeout = 30

# ========== 安全与连接 ==========
# 是否启用SSH主机密钥检查，测试环境可设为False，生产环境建议为True[reference:16]
host_key_checking = False

# 指定默认的SSH私钥文件路径[reference:17]
private_key_file = ~/.ssh/id_rsa

# ========== 性能与缓存 ==========
# 事实（Facts）收集策略，'smart'表示智能收集[reference:19]
gathering = smart

# 启用事实缓存，可大幅提升Playbook执行速度
fact_caching = jsonfile
# 缓存文件存放路径
fact_caching_connection = /tmp/ansible_fact_cache
# 缓存过期时间（秒）
fact_caching_timeout = 86400

# ========== 输出与日志 ==========
# 日志文件路径[reference:23]
log_path = ./ansible.log

# 控制台输出格式，'yaml'格式更易读[reference:25]
stdout_callback = yaml

# 是否显示废弃功能的警告[reference:27]
deprecation_warnings = False

[privilege_escalation]
# ========== 权限提升 ==========
# 是否启用权限提升[reference:28]
become = True
# 提权方式，默认为sudo[reference:29]
become_method = sudo
# 提权目标用户，默认为root[reference:30]
become_user = root
# 是否提示输入提权密码[reference:31]
become_ask_pass = False

[ssh_connection]
# ========== SSH连接优化 ==========
# 是否开启SSH管道化，可显著提升执行速度[reference:32]
pipelining = True
# 控制SSH多路复用连接，减少握手开销[reference:33]
ssh_args = -o ControlMaster=auto -o ControlPersist=60s

[inventory]
# ========== 清单插件 ==========
# 启用的主机清单插件[reference:34][reference:35]
enable_plugins = host_list, script, auto, yaml, ini
```



```bash
# ansible 2.10以上才可以用init命令
# 生成一份包含所有默认配置（但被注释掉）的示例文件
ansible-config init --disabled > ansible.cfg

# 如果想包含所有插件的配置，可以使用 -t all 参数
ansible-config init --disabled -t all > ansible.cfg
```



## 0.4 Ad-Hoc

`ansible` 命令行（即 Ad-Hoc 命令）与 `ansible-playbook` 命令（即 Playbook）

Ad-Hoc 命令适合执行一次性的、临时的快速任务；而 Playbook 则是将一系列复杂的操作编写成可重复执行、具有强大编排能力的“剧本”

| 对比维度       | Ad-Hoc 命令 (`ansible`)                                | Playbook (`ansible-playbook`)                                |
| :------------- | :----------------------------------------------------- | :----------------------------------------------------------- |
| **核心定位**   | 执行**单个、临时**的快速任务                           | **编排、管理**复杂的、重复性的任务                           |
| **可复用性**   | **不可复用**，命令执行完即消失                         | **持久化**保存为 `.yml` 文件，可反复执行和版本控制           |
| **任务复杂度** | 只能处理**一个**简单的任务                             | 可包含**多个** Play、Task，实现复杂的工作流                  |
| **执行控制**   | 控制能力弱，主要通过 `-f` 参数控制并发数               | 控制力强，可精细控制执行顺序、条件、依赖、错误处理等         |
| **适用场景**   | 临时重启服务、批量执行命令、快速查看系统信息、环境探测 | 应用部署、配置管理、系统初始化、CI/CD 流水线等标准化、重复性任务 |

## 0.5 Play-book

Ansible Playbook 使用 YAML 格式编写

- **文件开头**：文件通常以三个短横线 `---` 开头，表示 YAML 文档的开始。
- **注释**：使用 `#` 号进行注释。
- **键值对**：使用 `key: value` 的形式，**冒号后必须跟一个空格**。
- **列表/数组**：使用一个短横线加一个空格 `-` 表示列表项。
- **缩进**：使用**空格**进行缩进以表示层级关系，**不能使用 Tab 键**。同一层级的元素缩进必须一致。
- **大小写敏感**

一个标准的 Playbook 由以下几个核心部分组成：

1. **`name` (Play 的名称)**：可选但强烈建议添加，用于描述 Play 的目的，提高可读性。
2. **`hosts` (目标主机)**：**必选**。指定该 Play 要在哪些主机或主机组上执行。
3. **`remote_user` (远程用户)**：可选。指定在远程主机上执行任务的用户，默认为当前用户。
4. **`become` (权限提升)**：可选。布尔值，设为 `yes` 表示任务执行时使用 `sudo` 等方式提权。
5. **`vars` (变量)**：可选。在 Play 级别定义变量。
6. **`tasks` (任务列表)**：**必选**。一个有序列表，定义了该 Play 要执行的所有任务。
7. **`handlers` (处理器)**：可选。由任务通知触发的特殊任务列表，通常在 Play 的末尾执行。
8. **`roles` (角色)**：可选。用于组织和复用 Playbook 的一种高级方式

### 0.5.1 tasks

`tasks` 是一个列表，每个任务至少包含一个 `name`（任务名称）和一个要调用的**模块**

```yaml
tasks:
  - name: 安装最新版的 Apache
    ansible.builtin.yum:
      name: httpd
      state: latest
```

#### 常用模块

模块是 Ansible 的执行单元。一些常用的模块包括：

- **文件操作**：`ansible.builtin.file`、`ansible.builtin.copy`、`ansible.builtin.template`。
- **命令执行**：`ansible.builtin.command`、`ansible.builtin.shell`。
- **软件包管理**：`ansible.builtin.yum`/`dnf` (RedHat系)、`ansible.builtin.apt` (Debian系)。
- **服务管理**：`ansible.builtin.service` 或 `ansible.builtin.systemd`。
- **调试**：`ansible.builtin.debug`，用于输出信息，非常有助于调试

### 0.5.2 vars

```yaml
vars:
  apache_version: "2.4"

tasks:
  - name: 安装指定版本的 Apache
    ansible.builtin.yum:
      name: httpd-{{ apache_version }}
      state: present
```

### 0.5.3 when

使用 `when` 语句，可以根据条件决定是否执行某个任务

```yaml
tasks:
  - name: 在 CentOS 上安装 Apache
    ansible.builtin.yum:
      name: httpd
      state: present
    when: ansible_facts['os_family'] == "RedHat"
```

### 0.5.4 loop

使用 `loop` 关键字可以迭代一个列表，为列表中的每个元素执行一次任务

```yaml
tasks:
  - name: 创建多个用户
    ansible.builtin.user:
      name: "{{ item }}"
      state: present
    loop:
      - alice
      - bob
      - charlie
```

### 0.5.5 handlers

`handlers` 和任务类似，但只有在被任务通过 `notify` 通知时才会执行。常用于在配置更改后重启服务

```yaml
tasks:
  - name: 更新 Apache 配置文件
    ansible.builtin.template:
      src: httpd.conf.j2
      dest: /etc/httpd/conf/httpd.conf
    notify: 重启 Apache

handlers:
  - name: 重启 Apache
    ansible.builtin.service:
      name: httpd
      state: restarted
```

### 0.5.6 检查与执行

```bash
# 语法检查
ansible-playbook --syntax-check your-playbook.yml

# 执行
ansible-playbook -i inventory.ini your-playbook.yml
```

## 0.6 模块

ansible 通过-m指定模块

`-a` 参数用于 **向指定的模块传递执行所需的参数**。

### 0.6.1 command

默认没有-m 时，默认为command

**它的核心作用就是：在远程被控主机上执行一条 Shell 命令，但不经过远程主机的 Shell 环境（如 bash）处理。**

### `command` vs `shell`（关键区别）

| 对比维度                     | `-m command`                       | `-m shell`                           |
| :--------------------------- | :--------------------------------- | :----------------------------------- |
| **是否通过 Shell 执行**      | ❌ 不通过，直接调用可执行文件       | ✅ 通过 `/bin/sh` 执行                |
| **支持管道 `|`、重定向 `>`** | ❌ 不支持（会被当成普通参数）       | ✅ 支持                               |
| **支持环境变量 `$HOME`**     | ❌ 不支持（不会展开）               | ✅ 支持                               |
| **支持通配符 `\*`**          | ❌ 不支持                           | ✅ 支持                               |
| **安全性**                   | 更高（不易受 shell 注入攻击）      | 相对较低                             |
| **适用场景**                 | 简单的系统命令、不带特殊符号的脚本 | 复杂的组合命令、需要变量和管道的场景 |

# log

## 分享文件

```bash
ansible all -m copy -a "src=/本地/文件.txt dest=/远程/文件.txt"

# 指定 IP 地址
ansible 192.168.1.20 -m copy -a "src=... dest=..."

# 指定组名（批量操作组内所有主机）
ansible webservers -m copy -a "src=... dest=..."

# 匹配所有以 web 开头的主机
ansible web* -m copy -a "src=... dest=..."   

#  排除特定主机（除了某一台，其他都要）
ansible all,!web1 -m copy -a "src=... dest=..."   # 注意感叹号，表示排除 web1

# 1. 先创建目录
ansible all -m shell -a "mkdir -p ~/Downloads/"
# 2. 再拷贝文件
ansible all -m copy -a "src=~/jdk-...rpm dest=~/Downloads/"


# Ansible 版本 ≥ 2.10
# copy 模块支持 create_dirs 参数，设置为 yes 即可自动创建目标目录（包括缺失的父目录）：
ansible all -m copy -a "src=~/jdk-...rpm dest=~/Downloads/ create_dirs=yes"

```

 假设要把 `serverA` 上的 `/var/log/app.log` 拷贝到 `serverB` 的 `/tmp/` 目录下

```bash
ansible serverB -m ansible.posix.synchronize -a "src=/var/log/app.log dest=/tmp/ mode=pull" --delegate-to serverA
```

**工作原理：**

- `--delegate-to serverA` 表示这个命令的**执行动作**在 `serverA` 上发起。
- `mode=pull` 表示 `serverA` 从自己本地（src）拉取文件，然后推送给 `serverB`（dest）。

> ⚠️ **注意**：如果你的 Ansible 版本较旧（< 2.10），模块名可能是 `synchronize` 而不是 `ansible.posix.synchronize`。另外，目标主机上需要安装 `rsync` 命令。



## 需要输入密码的命令执行

关于“密码”的几种处理方式

| 场景                         | 操作方式                                           | 说明                                                         |
| :--------------------------- | :------------------------------------------------- | :----------------------------------------------------------- |
| **交互式输入**               | 命令行加 `-K`                                      | 最安全，每次执行时手动输入密码。                             |
| **免密配置（推荐）**         | 在被控机上配置 `sudoers` 的 `NOPASSWD`             | 最方便，适合自动化流水线。`visudo` 添加：`your_user ALL=(ALL) NOPASSWD: ALL` |
| **明文写入配置（极不推荐）** | 在 Inventory 中写 `ansible_become_password=123456` | 虽然能用，但严重安全隐患，**生产环境禁用**。                 |
| **使用 Vault 加密**          | `ansible-vault encrypt_string` 加密密码            | 企业级标准做法，将密码加密后放入 Inventory。                 |

免密配置配置方法

```bash
# 在被控主机上配置
# 最规范的做法是不要在 /etc/sudoers 主文件里改，而是在 /etc/sudoers.d/ 目录下新建一个专属文件。这样升级系统时不会被覆盖，管理也清晰。
visudo -f /etc/sudoers.d/ansible

# 添加内容
your_username ALL=(ALL) NOPASSWD: ALL
```

| 部分            | 含义                                                  |
| :-------------- | :---------------------------------------------------- |
| `your_username` | 指定对哪个用户生效（即你用来连接 Ansible 的系统账号） |
| `ALL=`          | 允许在**所有**主机上执行（通常保留这个）              |
| `(ALL)`         | 允许以**所有**用户的身份执行命令                      |
| `NOPASSWD:`     | **核心关键词**：无需密码                              |
| `ALL`           | 允许执行**所有**命令                                  |

### ansible

```bash
# 需要密码：如果目标主机的 sudo 需要输入密码，加上 -K（会提示你输入密码）。
ansible web1 -m command -a "systemctl restart nginx" -b -K
# 执行后会提示: BECOME password: 

# 免密 sudo：如果目标主机已配置 NOPASSWD，直接加 -b 即可。
ansible web1 -m copy -a "src=/tmp/hosts dest=/etc/hosts" -b

# 指定提权用户：如果想 sudo 成 root 以外的用户（比如 su - postgres），使用 --become-user
ansible web1 -m command -a "whoami" -b --become-user=postgres
# 输出结果将是 postgres
```

### ansible-playbook

```yaml
- name: 重启 nginx 服务
  hosts: web1
  become: yes                     # 等同于命令行的 -b
  become_user: root               # 可选，默认为 root，等同于 --become-user
  tasks:
    - name: 重启服务
      systemd:
        name: nginx
        state: restarted
```

```bash
ansible-playbook restart_nginx.yml -K
```

### 全局配置

在ansible.cfg中

```bash
[privilege_escalation]
become = True
become_method = sudo
become_user = root
become_ask_pass = False    # 如果免密则设为 False，否则设为 True
```

**在 Inventory（主机清单）中**（针对特定主机）

```bash
[webservers]
web1 ansible_host=192.168.1.10 ansible_become=true ansible_become_user=root
```



## 主控也加入执行相同程序

```bash
[all:children]
control
managed

[control]
localhost ansible_connection=local   # 使用 local 连接，避免 SSH 自身

[managed]
web1 ansible_host=192.168.1.10
web2 ansible_host=192.168.1.20
```

