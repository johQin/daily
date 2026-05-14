# Claude Code

# 1 安装配置

```bash
# 安装
C:\Users\35900>winget install Anthropic.ClaudeCode
The `msstore` source requires that you view the following agreements before using.
Terms of Transaction: https://aka.ms/microsoft-store-terms-of-transaction
The source requires the current machine's 2-letter geographic region to be sent to the backend service to function properly (ex. "US").

Do you agree to all the source agreements terms?
[Y] Yes  [N] No: Y
Found Claude Code [Anthropic.ClaudeCode] Version 2.1.126
This application is licensed to you by its owner.
Microsoft is not responsible for, nor does it grant any licenses to, third-party packages.
Downloading https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/2.1.126/win32-x64/claude.exe
  ██████████████████████████████   234 MB /  234 MB
Successfully verified installer hash
Starting package install...
Path environment variable modified; restart your shell to use the new value.
Command line alias added: "claude"
Successfully installed

# 版本查看
claude -v
2.1.126 (Claude Code)

# 使用cmd中，claude命令进入对话，如果出现下面的文字，则账号有问题
# Not logged in · Run /login
# 法一：使用官方账号，在控制台直接输入 /login 并按回车。它会弹出一个网页，你登录你的 Anthropic (Claude.ai) 账号并授权即可。
# 法二：登录欺骗
echo '{"hasCompletedOnboarding": true}' > ~/.claude.json
echo '{"primaryApiKey": "any-string"}' > ~/.claude/config.json

# 使用本地代理，配置多个国内模型，可以在claude对话里通过/model切换
# Claude Code Router (CCR) → 最强大方式（支持运行中动态切换）
# https://musistudio.github.io/claude-code-router/zh-CN/
npm install -g claude-code-router

```

# 2 [claude-code-router]( https://musistudio.github.io/claude-code-router/zh-CN/)

```bash
npm install -g @musistudio/claude-code-router
```

## 2.1 Command

#### start

```bash
ccr start [选项]

# 监听端口
# --port number
# -p

# 配置文件路径
# --config path
# -c

# 作为守护进程运行（后台进程）
# --daemon
# -d

# 日志级别
# --log-level <level | fatal | error | warn | info | debug | trace>
# -l


```

#### model

```bash
# 交互式选择模型
ccr model
# 这将显示一个包含可用提供商和模型的交互式菜单。
# 功能：创建新提供商，向现有提供商添加新模型

# 设置默认模型
ccr model set <provider>,<model>
# eg:
ccr model set deepseek,deepseek-chat

# 列出所有配置的模型
ccr model list

# 添加模型
ccr model add <provider>,<model>

# 从配置中删除模型
ccr model remove <provider>,<model>
```



#### status

```bash
ccr status
```

#### preset

管理预设（presets）——可共享和重用的配置模板

预设功能让您可以：

- 将当前配置保存为可重用的模板
- 与他人分享配置
- 安装社区提供的预配置方案
- 在不同配置之间轻松切换

```bash
# 将当前配置导出为预设。
ccr preset export <名称> [选项]

# --output _path 自定义输出目录路径 
# --description _desc 预设描述
# --author _author 预设作者
# --tag _tag 逗号分隔的关键字
# --include-sensitive - 包含 API 密钥等敏感数据（不推荐）s

# eg: 
ccr preset export my-config --description "我的生产环境配置" --author "您的名字"
# 执行过程
# 1. 读取 ~/.claude-code-router/config.json 中的当前配置
# 2. 如未通过命令行提供，提示输入描述、作者和关键字
# 3. 自动清理敏感字段（API 密钥变为占位符）
# 4. 在 ~/.claude-code-router/presets/<名称>/ 创建预设目录，生成包含配置和元数据的 manifest.json

ccr preset install <source>

# 从目录安装
ccr preset install ./my-preset
# 重新配置已安装的预设
ccr preset install my-preset

# 执行过程
# 1. 从预设目录读取 manifest.json
# 2. 验证预设结构
# 3. 如果预设包含 schema，提示输入必需的值（API 密钥等）
# 4. 将预设复制到 ~/.claude-code-router/presets/<名称>/
# 5. 在 manifest.json 中保存用户输入

# 列出所有已安装的预设。
ccr preset list

# Available presets:

# • my-config (v1.0.0)
#   My production setup
#   by Your Name

# • openai-setup
#  Basic OpenAI configuration

# 显示预设的详细信息
ccr preset info <名称>

# 删除已安装的预设
ccr preset delete <名称>
ccr preset rm <名称>
ccr preset remove <名称>

```

预设的结构

预设是一个包含 `manifest.json` 文件的目录

```json
{
  "name": "my-preset",
  "version": "1.0.0",
  "description": "我的配置",
  "author": "作者姓名",
  "keywords": ["openai", "production"],

  "Providers": [
    {
      "name": "openai",
      "api_base_url": "https://api.openai.com/v1/chat/completions",
      "api_key": "{{apiKey}}",
      "models": ["gpt-4", "gpt-3.5-turbo"]
    }
  ],

  "Router": {
    "default": "openai,gpt-4"
  },

  "schema": [
    {
      "id": "apiKey",
      "type": "password",
      "label": "OpenAI API 密钥",
      "prompt": "请输入您的 OpenAI API 密钥"
    }
  ]
}
```

Schema 列表字段里的每一项，定义了用户在安装时必须提供的输入，这些字段的值可以是下列类型：

1. `password` - 隐藏输入（用于 API 密钥）
2. `input` - 文本输入
3. `select` - 单选下拉框
4. `multiselect` - 多选下拉框
5. `confirm` - 是/否确认
6. `editor` - 多行文本编辑器
7. `number` - 数字输入

默认情况下，`export` 会清理敏感字段：

- 名为 `api_key`、`apikey`、`password`、`secret` 的字段会被替换为 `{{字段名}}` 占位符
- 这些占位符会成为 schema 中的必需输入
- 用户在安装时会被提示提供自己的值

#### 其他命令

```bash
ccr stop
ccr restart

# 通过路由器执行 claude 命令
ccr code [参数...]

# 在浏览器中打开 Web UI。
ccr ui
# http://127.0.0.1:3456/ui/
```

## 2.2 配置

存在两级的配置：

- 全局配置
- 项目级配置
- 

# 2 核心工作流

### 2.1 工作模式

1. Plan
   - 让Claude 只规划，不执行，只读取文件理解文本，不修改任何文件，反复讨论，修改方案
   - Shift + Tab + Tab 进入 Plan模式
   - Shift + Tab 切换到正常模式
2. Auto
   - 用一个AI分类器替你做权限判断。安全操作自动放行，危险操作才拦截
   - 
3. 

