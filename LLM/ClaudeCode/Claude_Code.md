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

# 4 [.claude目录](https://code.claude.com/docs/en/claude-directory)

.claude的位置——Claude Code 读取 CLAUDE.md、settings.json、hooks、skills、commands、subagents、workflows、rules 和自动内存的位置。

项目文件内的`.claude`文件夹，可以提交到git分享给团队，而放置到`~/.claude`中的文件是个人配置，适用于你的所有项目。

![image-20260622163613706](legend/image-20260622163613706.png)

大多数用户只编辑 `CLAUDE.md` 和 `settings.json`，目录的其余部分是可选的。

## 4.1 作用域与优先级

### 4.1.1 作用域

| 作用域      | 位置                                                         | 影响范围                                                     | 与团队共享？     |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------------- |
| **Managed** | Anthropic 服务器管理的设置、plist / 注册表或系统级 `managed-settings.json` | 服务器管理交付的所有组织成员；plist、HKLM 注册表和文件交付的机器上的所有用户；HKCU 注册表交付的当前用户 | 是（由 IT 部署） |
| **User**    | `~/.claude/` 目录                                            | 您，跨所有项目                                               | 否               |
| **Project** | 存储库中的 `.claude/`                                        | 此存储库上的所有协作者                                       | 是（提交到 git） |
| **Local**   | `.claude/settings.local.json`                                | 您，仅在此存储库中                                           | 否（gitignored） |



作用域适用于许多 Claude Code 功能：

| 功能            | User 位置                 | Project 位置                       | Local 位置                    |
| :-------------- | :------------------------ | :--------------------------------- | :---------------------------- |
| **Settings**    | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json` |
| **Subagents**   | `~/.claude/agents/`       | `.claude/agents/`                  | 无                            |
| **MCP servers** | `~/.claude.json`          | `.mcp.json`                        | `~/.claude.json`（每个项目）  |
| **Plugins**     | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json` |
| **CLAUDE.md**   | `~/.claude/CLAUDE.md`     | `CLAUDE.md` 或 `.claude/CLAUDE.md` | `CLAUDE.local.md`             |

### 4.1.2 优先级

1. **Managed**（最高）- 无法被任何内容覆盖
2. **命令行参数** - 临时会话覆盖
3. **Local** - 覆盖项目和用户设置
4. **Project** - 覆盖用户设置
5. **User**（最低）- 当没有其他内容指定设置时应用

## 4.2 [CLAUDE.md](https://code.claude.com/docs/en/memory)

**When it loads**：Loaded into context at the start of every session

这个文件可以跨session，instruction（指示）claude的行为。

用于指示claude怎么工作，请在这里放置你们的惯例、常用命令和架构背景，这样Claude就能与你们的团队持有相同的假设进行操作。

- `CLAUDE.md`希望可以少于200行。他的所有内容都会被全部加载，太长的话会费token，并且会降低它的约束力。
- `.claude/CLAUDE.md`在每次session的一开始就会被加载，列出你最常运行的命令，例如 build、test 和 format，这样 Claude 就不需要你每次都拼写出来

- 如果某事只对特定任务有关系，就把它移动到skills或rules中，以便在需要时才加载
- 在某次session中，可以运行`/memory`，可以打开和编辑`CLAUDE.md`（临时性）
- 如果希望项目根目录的整洁度，可以将项目根目录的`CLAUDE.md`放置到`.claude/CLAUDE.md`，这同样生效。



## 4.3 [settings.json](https://code.claude.com/docs/zh-CN/settings)

项目中的 `.claude/settings.json` 会与用户级（`~/.claude/settings.json`）的配置进行**对象深度合并**，而不是完全覆盖。

**如果项目配置中没有配置 A，而用户级配置中有 A，那么 A 会保留用户配置的值，继续生效。**

合并方式是 **递归覆盖（Recursive Merge）**：

- 对于**普通字段**（字符串、数字、布尔值）：项目级有值 → 用项目级；项目级没有 → 用用户级。
- 对于**对象（Object）**：会递归合并。项目级对象中的字段会覆盖用户级同名字段，但用户级中项目级没有的字段会保留。
- 对于**数组（Array）**：**通常直接替换**，而不是合并。项目级的数组会完全覆盖用户级的数组（这一点需要注意，不是追加）。

`settings.local.json` ——git会忽略，`settings.json`——git不会忽略，而分享给team

### 4.3.1 编辑何时生效

Claude Code 监视您的设置文件，并在它们更改时重新加载它们，因此对大多数键的编辑会在运行的会话中应用，无需重启。这包括 `permissions`、`hooks` 和凭证助手（如 `apiKeyHelper`）。重新加载涵盖用户、项目、本地和 managed 设置，并为每个检测到的更改触发 [`ConfigChange` hook](https://code.claude.com/docs/zh-CN/hooks#configchange)。

少数几个键在会话启动时读取一次，并在下次重启时应用：

- `model`：使用 [`/model`](https://code.claude.com/docs/zh-CN/model-config#setting-your-model) 在会话中切换
- [`outputStyle`](https://code.claude.com/docs/zh-CN/output-styles)：系统提示的一部分，在 `/clear` 或重启时重建

### 4.3.2 常见配置

#### 4.3.2.1 permissions

`permissions` 是用来控制 Claude Code 操作权限的“总开关”。它定义了 Claude 在什么情况下需要向你请求批准，什么情况下可以自主行动。

- allow：允许工具使用的**权限规则**数组，**[权限规则语法](https://code.claude.com/docs/zh-CN/settings#permission-rule-syntax)在下面会讲到**
- ask：在工具使用时要求 询问确认 的权限规则数组
- deny：拒绝工具使用的权限规则数组。
- additionalDirectories：Claude 有权访问的额外工作目录
- defaultMode
- disableBypassPermissionsMode：设置为 `"disable"` 以防止激活 `bypassPermissions` 模式。
- skipDangerousModePermissionPrompt：跳过通过 `--dangerously-skip-permissions` 或 `defaultMode: "bypassPermissions"` 进入 bypass permissions 模式前显示的确认提示。



##### defaultMode

而 `defaultMode` 就是设定这个“总开关”的默认档位，它决定了 Claude 在**每次会话开始时**的行为基调。你可以把它想象成一个“工作模式”选择器。

`defaultMode`的可选值

- default or manual：**标准/手动模式**。Claude 只在首次使用某个需要批准的工具时询问你，后续操作（如文件读写）不再重复提问
- **`acceptEdits`**：**自动批准编辑**。Claude 可以自动批准对工作目录内文件的编辑，以及 `mkdir`、`touch`、`mv`、`cp` 等常见文件操作，无需你逐个点击“允许
- **`plan`**：**计划模式**。Claude **只会读取文件**和运行只读命令来探索和分析代码，**不会**对你的源代码进行任何编辑。它纯粹用于制定计划
- **`bypassPermissions`**：**绕过所有权限**。Claude **跳过几乎所有权限提示**，可以自由行动（除了个别强制提示的风险操作）
- **`auto`**：**自动模式**。Claude 自动批准大部分工具调用，但后台会有安全检查，确保操作与你的请求意图一致
- **`dontAsk`**：**不要询问**。Claude 会**自动拒绝**所有未被 `permissions.allow` 规则明确预先批准的的工具调用

#### 4.3.2.2 hooks

配置自定义命令以在生命周期事件处运行

事件分为三种频率：

- 每个会话一次：`SessionStart` 和 `SessionEnd`
- 每轮一次：`UserPromptSubmit`、`Stop` 和 `StopFailure`
- 代理循环内的每个工具调用：`PreToolUse` 和 `PostToolUse`

![Hook 生命周期图，显示可选的 Setup 流入 SessionStart，然后是每轮循环，包含 UserPromptSubmit、用于 slash commands 的 UserPromptExpansion、嵌套的代理循环（PreToolUse、PermissionRequest、PostToolUse、PostToolUseFailure、PostToolBatch、SubagentStart/Stop、TaskCreated、TaskCompleted）和 Stop 或 StopFailure，然后是 TeammateIdle、PreCompact、PostCompact 和 SessionEnd，Elicitation 和 ElicitationResult 嵌套在 MCP 工具执行内，PermissionDenied 作为 PermissionRequest 的副分支用于自动模式拒绝，WorktreeCreate、WorktreeRemove、Notification、ConfigChange、InstructionsLoaded、CwdChanged 和 FileChanged 作为独立异步事件，MessageDisplay 作为仅显示事件，在助手消息文本流式传输时运行](legend/hooks-lifecycle.svg)

| 事件                | 触发时机                                                     |
| :------------------ | :----------------------------------------------------------- |
| SessionStart        | 当会话开始或恢复时                                           |
| Setup               | 当你使用 `--init-only` 启动 Claude Code，或在 `-p` 模式下使用 `--init` 或 `--maintenance` 时。用于 CI 或脚本中的一次性准备 |
| UserPromptSubmit    | 当你提交提示词时，在 Claude 处理它之前                       |
| UserPromptExpansion | 当用户输入的命令扩展为提示词时，在到达 Claude 之前。可以阻止该扩展 |
| PreToolUse          | 在工具调用执行之前。可以阻止它                               |
| PermissionRequest   | 当权限对话框出现时                                           |
| PermissionDenied    | 当自动模式分类器拒绝工具调用时。返回 `{retry: true}` 可告知模型它可以重试被拒绝的工具调用 |
| PostToolUse         | 在工具调用成功后                                             |
| PostToolUseFailure  | 在工具调用失败后                                             |
| PostToolBatch       | 在一批并行工具调用全部解析完毕后，在下一次模型调用之前       |
| Notification        | 当 Claude Code 发送通知时                                    |
| MessageDisplay      | 当助手消息文本正在显示时                                     |
| SubagentStart       | 当子代理被生成时                                             |
| SubagentStop        | 当子代理完成时                                               |
| TaskCreated         | 当通过 TaskCreate 创建任务时                                 |
| TaskCompleted       | 当任务被标记为已完成时                                       |
| Stop                | 当 Claude 完成响应时                                         |
| StopFailure         | 当本轮对话因 API 错误而结束时。输出和退出代码将被忽略        |
| TeammateIdle        | 当代理团队中的某个队友即将进入空闲状态时                     |
| InstructionsLoaded  | 当 CLAUDE.md 或 `.claude/rules/*.md` 文件被加载到上下文中时。在会话开始时以及会话期间按需加载文件时触发 |
| ConfigChange        | 当会话期间配置文件发生变化时                                 |
| CwdChanged          | 当工作目录发生变化时，例如 Claude 执行 `cd` 命令时。对于使用 direnv 等工具进行响应式环境管理很有用 |
| FileChanged         | 当磁盘上的被监视文件发生变化时。`matcher` 字段指定要监视哪些文件名 |
| WorktreeCreate      | 当通过 `--worktree`、`isolation: "worktree"` 或为后台会话创建工作树时。替换默认的 git 行为 |
| WorktreeRemove      | 当会话退出时、子代理完成时或你删除后台会话时移除工作树       |
| PreCompact          | 在上下文压缩之前                                             |
| PostCompact         | 在上下文压缩完成之后                                         |
| Elicitation         | 当 MCP 服务器在工具调用期间请求用户输入时                    |
| ElicitationResult   | 在用户响应 MCP 提示后，在将响应发送回服务器之前              |
| SessionEnd          | 当会话终止时                                                 |

#### 4.3.2.3 [statusLine](https://code.claude.com/docs/zh-CN/statusline)

配置自定义状态栏以监控 Claude Code 中的上下文窗口使用情况、成本和 git 状态

```json
{

    "statusLine": {
        "type": "command",
        "command": "bash ./.claude/statusline.sh"
    }

}
```



```bash
#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
INPUT=$(echo "$input" | jq -r '.context_window.total_input_tokens // 0')
OUTPUT=$(echo "$input" | jq -r '.context_window.total_output_tokens // 0')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
EFFORT=$(echo "$input" | jq -r '.effort.level // "UNK"')

echo "$MODEL | in:$INPUT out:$OUTPUT pct:${PCT}% eft:$EFFORT"

# jq需要提前安装，https://github.com/jqlang/jq/releases
# 或者 winget install jqlang.jq
# 安装后添加PATH
# 检测：jq --version
```



## 4.4 [rules/*.md](https://code.claude.com/docs/en/memory#organize-rules-with-claude/rules/)

**When it loads**：

| 规则类型                  | 首次加载时机             | 上下文压缩后             |
| :------------------------ | :----------------------- | :----------------------- |
| **`*.md`无 `paths` 字段** | 会话启动时               | 自动重新注入             |
| **`*.md`有 `paths` 字段** | **意图**：操作匹配文件时 | 丢失，匹配文件时重新加载 |

```markdown
---
paths:
  - "**/*.test.ts"
  - "**/*.test.tsx"
---

# Testing Rules

- Use descriptive test names: "should [expected] when [condition]"
- Mock external dependencies, not internal modules
- Clean up side effects in afterEach
```



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
      "api_key": "sk-imbvmqlmjxxxxxx",
      "models": [
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "Pro/zai-org/GLM-5.1",
        "Qwen/Qwen3.5-397B-A17B"
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
      "api_key": "sk-1a8689ec91da4xxxx",
      "models": [
        "deepseek-v4-flash",
        "deepseek-v4-pro"
      ],
      "transformer": {
        "use": [
          "deepseek"
        ]
      }
    },
    {
      "name": "volcengine",
      "api_base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
      "api_key": "ark-57ba57e6-0b2xxxx",
      "models": [
        "doubao-seed-2-0-lite-260428",
        "doubao-seed-2-0-pro-260215"
      ],
      "transformer": {
        "use": [
          "doubao"
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
    "image": "siliconflow,Qwen/Qwen3.5-397B-A17B"
  },
  "CUSTOM_ROUTER_PATH": ""
}
```

```bash
# 切换模型
/model siliconflow,Pro/zai-org/GLM-5.1
/model siliconflow,deepseek-ai/DeepSeek-V4-Pro
/model siliconflow,deepseek-ai/DeepSeek-V4-Flash
/model deepseek,deepseek-v4-pro
/model volcengine,doubao-seed-2-0-lite-260428

https://ark.cn-beijing.volces.com/api/v3/chat/completions
# 火山平台测试
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions -H "Authorization: Bearer ark-57ba57e6-0b20-40fd-bb21-d9cfcb899df4-5659a" -H "Content-Type: application/json" -d '{
  "model":"doubao-seed-2-0-lite-260428",
  "messages":[{"role":"user","content":"test"}]
}'


# 输出包含按模型拆分的完整 Token 消耗
/cost
/usage
#会话 token 总量、模型使用频次、上下文占用、套餐剩余额度
/stats

# 定位 Token 消耗来源，但会展示哪些文件 / 对话历史 / MCP 工具吃掉大量 token，用于优化上下文减少消耗
/context

# 状态栏实时显示 Token（常驻监控）
/config status_line true

# 压缩上下文
/compact

/d/repos/dikong/scdk/20260720
```

image_rId22.png

[claude code 命令大全1](https://zhuanlan.zhihu.com/p/2020457076900537879)

[claude code 命令大全2](https://blog.csdn.net/weixin_56693899/article/details/161024592)



```
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions \
-H "Authorization: Bearer ark-57ba57e6-0b20-40fd-bb21-d9cfcb899df4-5659a" \
-H "Content-Type: application/json" \
-d '{
  "model":"doubao-seed-2-0-lite-260428",
  "messages":[{"role":"user","content":"test"}]
}'
