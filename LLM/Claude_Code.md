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

## claude-code-router

```bash
npm install -g @musistudio/claude-code-router


```

### 命令

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

