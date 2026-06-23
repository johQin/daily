# Claude Code

# 1 安装配置

## 安装claude code

```bash
# 法一(推荐)：node npm 安装
npm install -g @anthropic-ai/claude-code
# 法二：powershell
winget install Anthropic.ClaudeCode

# 版本查看
claude -v
2.1.126 (Claude Code)

# 使用本地代理，配置多个国内模型，可以在claude对话里通过/model切换
# Claude Code Router (CCR) → 最强大方式（支持运行中动态切换）
# https://musistudio.github.io/claude-code-router/zh-CN/
npm install -g claude-code-router

# 使用ccr code进入claude code 交互命令窗
ccr code
# 如果你直接使用claude 命令进入交互命令窗，可能会出现下面的问题
# Not logged in · Run /login
# 法一：使用官方账号，在控制台直接输入 /login 并按回车。它会弹出一个网页，你登录你的 Anthropic (Claude.ai) 账号并授权即可。
# 法二：登录欺骗
echo '{"hasCompletedOnboarding": true}' > ~/.claude.json
echo '{"primaryApiKey": "any-string"}' > ~/.claude/config.json
```

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

# 3 扩展claude的功能



**初次使用 Claude Code？** 从 CLAUDE.md开始了解项目约定，然后根据需要添加其他扩展当特定触发器出现时。

| 功能                                                         | 作用                                         | 何时使用                                           | 示例                                                      |
| :----------------------------------------------------------- | :------------------------------------------- | :------------------------------------------------- | :-------------------------------------------------------- |
| [**CLAUDE.md**](https://code.claude.com/docs/zh-CN/memory)   | 每次对话加载的持久上下文                     | 项目约定、“始终执行 X” 规则                        | ”使用 pnpm，而不是 npm。提交前运行测试。“                 |
| [**Skill**](https://code.claude.com/docs/zh-CN/skills)       | Claude 可以使用的说明、知识和工作流          | 可重用内容、参考文档、可重复的任务                 | `/deploy` 运行您的部署清单；包含端点模式的 API 文档 skill |
| [**Subagent**](https://code.claude.com/docs/zh-CN/sub-agents) | 返回摘要结果的隔离执行上下文                 | 上下文隔离、并行任务、专门的工作者                 | 读取许多文件但仅返回关键发现的研究任务                    |
| [**Agent teams**](https://code.claude.com/docs/zh-CN/agent-teams) | 协调多个独立的 Claude Code 会话              | 并行研究、新功能开发、使用竞争假设进行调试         | 生成审查者同时检查安全性、性能和测试                      |
| **Code intelligence**                                        | 语言服务器导航和诊断                         | 类型化语言、大型代码库（其中 grep 速度慢或不精确） | 跳转到符号的定义，而不是读取整个文件                      |
| [**MCP**](https://code.claude.com/docs/zh-CN/mcp)            | 连接到外部服务                               | 外部数据或操作                                     | 查询您的数据库、发布到 Slack、控制浏览器                  |
| [**Hook**](https://code.claude.com/docs/zh-CN/hooks-guide)   | 由事件触发的脚本、HTTP 请求、提示或 subagent | 必须在每个匹配事件上运行的自动化                   | 每次文件编辑后运行 ESLint                                 |
| [Plugin](https://code.claude.com/docs/zh-CN/plugins)         | 捆绑和共享功能集                             |                                                    |                                                           |
| [Marketplaces](https://code.claude.com/docs/zh-CN/plugin-marketplaces) | 托管和分发plugin集合                         |                                                    |                                                           |

| 触发器                                         | 添加                                                         |
| :--------------------------------------------- | :----------------------------------------------------------- |
| Claude 两次出错约定或命令                      | 将其添加到 [CLAUDE.md](https://code.claude.com/docs/zh-CN/memory) |
| 您一直在输入相同的提示来启动任务               | 将其保存为用户可调用的 [skill](https://code.claude.com/docs/zh-CN/skills) |
| 您第三次将相同的剧本或多步骤过程粘贴到聊天中   | 将其捕获为 [skill](https://code.claude.com/docs/zh-CN/skills) |
| 您一直在从 Claude 看不到的浏览器标签页复制数据 | 将该系统连接为 [MCP server](https://code.claude.com/docs/zh-CN/mcp) |
| Claude 读取许多文件以查找符号的定义或使用位置  | 为您的语言安装 [code intelligence plugin](https://code.claude.com/docs/zh-CN/discover-plugins#code-intelligence) |
| 一个辅助任务用您不会再次引用的输出淹没您的对话 | 通过 [subagent](https://code.claude.com/docs/zh-CN/sub-agents) 路由它 |
| 您希望每次都发生某事而无需询问                 | 编写 [hook](https://code.claude.com/docs/zh-CN/hooks-guide)  |
| 第二个存储库需要相同的设置                     | 将其打包为 [plugin](https://code.claude.com/docs/zh-CN/plugins) |

## 3.1 功能比较

### 3.1.1 SubAgent vs Agent team

| 方面         | Subagent                           | Agent team                             |
| :----------- | :--------------------------------- | :------------------------------------- |
| **上下文**   | 自己的上下文窗口；结果返回给调用者 | 自己的上下文窗口；完全独立             |
| **通信**     | 仅向主代理报告结果                 | 队友直接相互发送消息                   |
| **协调**     | 主代理管理所有工作                 | 具有自我协调的共享任务列表             |
| **最适合**   | 仅结果重要的专注任务               | 需要讨论和协作的复杂工作               |
| **令牌成本** | 较低：结果摘要返回到主上下文       | 较高：每个队友是一个单独的 Claude 实例 |

## 3.2 分层定义功能（多级别）

功能可以在多个级别定义：用户范围、每个项目、通过 plugins 或通过托管策略。

## 3.3 功能的上下文消耗

您添加的每个功能都会消耗 Claude 的一些上下文

每个功能都有不同的加载策略和上下文成本：

| 功能                  | 何时加载          | 加载内容                           | 上下文成本                   |
| :-------------------- | :---------------- | :--------------------------------- | :--------------------------- |
| **CLAUDE.md**         | 会话开始          | 完整内容                           | 每个请求                     |
| **Skills**            | 会话开始 + 使用时 | 启动时的描述，使用时的完整内容     | 低（每个请求的描述）*        |
| **MCP 服务器**        | 会话开始          | 工具名称；完整架构按需             | 低，直到使用工具             |
| **Code intelligence** | 文件编辑后和按需  | 编辑后的诊断；符号查找时的位置信息 | 低；减少其他地方的文件读取   |
| **Subagents**         | 生成时            | 具有指定 skills 的新鲜上下文       | 与主会话隔离                 |
| **Hooks**             | 触发时            | 无（外部运行）                     | 零，除非 hook 返回额外上下文 |

# 4 .claude目录

.claude的位置——Claude Code 读取 CLAUDE.md、settings.json、hooks、skills、commands、subagents、workflows、rules 和自动内存的位置

![image-20260622163613706](legend/image-20260622163613706.png)

# 其他内容-----------------------------------------------------------

# 1 [claude-code-router]( https://musistudio.github.io/claude-code-router/zh-CN/)

```bash
npm install -g @musistudio/claude-code-router
```

## 1.1 Command

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

## 1.2 配置

存在两级的配置：

- 全局配置
- 项目级配置
- 

```json
{
  "LOG": false,
  "LOG_LEVEL": "debug",
  "CLAUDE_PATH": "",
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "APIKEY": "",
  "API_TIMEOUT_MS": "600000",
  "PROXY_URL": "",
  "transformers": [],
  "Providers": [
    {
      "name": "siliconflow",
      "api_base_url": "https://api.siliconflow.cn/v1/chat/completions",
      "api_key": "sk-ixxxxxx",
      "models": [
        "deepseek-ai/DeepSeek-V4-Flash",
        "Pro/zai-org/GLM-5.1"
      ],
      "transformer": {
        "use": [
          [
            "maxtoken",
            {
              "max_tokens": 16384
            }
          ]
        ]
      }
    },
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "sk-1axxxxx",
      "models": [
        "deepseek-v4-flash",
        "deepseek-v4-pro"
      ],
      "transformer": {
        "use": [
          "deepseek"
        ]
      }
    }
  ],
  "StatusLine": {
    "enabled": false,
    "currentStyle": "default",
    "default": {
      "modules": []
    },
    "powerline": {
      "modules": []
    }
  },
  "Router": {
    "default": "deepseek,deepseek-v4-flash",
    "background": "siliconflow,deepseek-ai/DeepSeek-V4-Flash",
    "think": "siliconflow,Pro/zai-org/GLM-5.1",
    "longContext": "",
    "longContextThreshold": 60000,
    "webSearch": "",
    "image": ""
  },
  "CUSTOM_ROUTER_PATH": ""
}
```

