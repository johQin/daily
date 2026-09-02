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

