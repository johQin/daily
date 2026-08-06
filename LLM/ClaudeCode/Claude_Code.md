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
2.1.218 (Claude Code)

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

### 4.1.2 优先级（precedence）

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

`/status`：可以看见当前session中哪些配置被激活。在status tab包含配置源（User settings、Project settings等）

### 4.3.2 常见配置

- permissions
- hooks
- statusLine
- model：设置一个默认的模型
- env：设置环境变量，以控制
- outputStyle

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

#### 4.3.2.4 [env](https://code.claude.com/docs/en/env-vars)

环境变量，可以用于控制claude code的行为，也可以设置环境变量以供命令行命令使用。

改变了env的值，要重启claude code才能生效。

```json
{
  "env": {
    "API_TIMEOUT_MS": "1200000",
    "BASH_DEFAULT_TIMEOUT_MS": "300000"
  }
}
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

## 4.5 skills

本质：**您或者claude可以通过名称调用的，可重用的，prompts，**

每一个skill是一个文件夹，其包含一个skill.md和skill引用的其他文件。

Skills扩展了Claude的能力。

Claude Code skills 遵循 [Agent Skills](https://agentskills.io/) 开放标准，该标准适用于多个 AI 工具。Claude Code 使用额外功能扩展了该标准，如[调用控制](https://code.claude.com/docs/zh-CN/skills#control-who-invokes-a-skill)、[subagent 执行](https://code.claude.com/docs/zh-CN/skills#run-skills-in-a-subagent)和[动态上下文注入](https://code.claude.com/docs/zh-CN/skills#inject-dynamic-context)。

**自定义命令已合并到 skills 中。** `.claude/commands/deploy.md` 中的文件和 `.claude/skills/deploy/SKILL.md` 中的 skill 都会创建 `/deploy` 并以相同的方式工作。

### 4.5.1 When it loads

**When it loads**：通过/skill-name指定调用某个skill，或者当claude 匹配到一个任务给skill时。

Claude Code 的 skill 相关信息并非一次性全部注入，而是分**两个阶段**注入到模型的上下文中

- 会话启动时——只注入“技能列表”
  - 在会话一开始，Claude 的上下文里就会加载一份所有可用技能的“名录”。这份列表包含了每个技能的**名称**和一段简短的**描述**
  - **目的**：让 Claude 知道有哪些技能可用，并能根据你的问题判断是否需要调用某个技能
  - **实现方式**：这份列表会通过一个`<system-reminder>`消息注入，帮助模型在看到相关问题时匹配到正确的技能
- 技能被调用时——注入“完整内容”
  - 当你通过`/skill-name`手动触发，或者 Claude 通过`SkillTool`自动决定使用某个技能时，该技能的核心内容才会被注入。
  - 这个阶段注入的**完整内容**包括：
    - **SKILL.md 的正文**：包含具体的指令和工作流程。
    - **动态上下文**：技能内通过 `!`` 语法包裹的 shell 命令会在此时**立刻执行**，执行结果会替换掉命令本身，再一起发送给模型

### 4.5.2 [bundled skills](https://code.claude.com/docs/zh-CN/commands)

在英文中，"built-in"和"bundled"的核心区别在于**"与生俱来"**和**"打包附赠"**。

[Claude Code 中可用命令的完整参考，包括内置命令和捆绑的 skills。](https://code.claude.com/docs/zh-CN/commands)

**Built-in** 更像是你汽车方向盘上的**物理按钮**，按下去就执行固定的功能；**Bundled** 则像是随车附赠的一本**高级驾驶指南**，AI 会阅读它，并根据当前路况（你的代码）来决定如何操作。

| 对比维度     | **Built-in Commands (内置命令)**                             | **Bundled Skills (捆绑技能)**                                |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **本质**     | **硬编码的逻辑**。由 CLI 的 TypeScript 代码直接执行，逻辑是固定写死的。 | **基于提示词（Prompt）的工作流**。本质是给 Claude 的一份详细"操作手册"，它依靠 Claude 的理解能力和工具来完成任务。 |
| **灵活性**   | **固定**，执行预定义的操作（如切换模型 `/model`、清空上下文 `/clear`）。 | **动态且自适应**，Claude 可以根据你的代码库情况灵活调整执行步骤。 |
| **可覆盖性** | **无法**被用户自定义的 skill 覆盖。                          | **可以**被覆盖。如果你在项目中创建了同名的 `code-review` skill，它会替换掉捆绑的版本。 |
| **可用性**   | 通常**无法禁用**（如核心命令）。                             | 可以通过设置 `disableBundledSkills: true` 来**统一禁用**所有捆绑技能。 |
| **典型例子** | `/help`、`/compact`、`/clear`、`/model` 等管理类命令。       | `/code-review`、`/debug`、`/batch`、`/claude-api` 等工作流类工具 |

### 4.5.3 自定义skill

#### 自定义示例

此示例创建一个 skill，用于总结你的 git 仓库中未提交的更改，并标记任何风险的内容。它在 Claude 读取之前将实时 diff 拉入提示中，因此响应基于你的实际工作树，而不是 Claude 从打开的文件中猜测的内容。当你询问你的更改时，Claude 会自动加载该 skill，或者你可以使用 `/summarize-changes` 直接调用它。

- 创建skill目录

  ```bash
  mkdir -p ~/.claude/skills/summarize-changes
  ```

- 编写skill.md

  - 每个 skill 都需要一个 `SKILL.md` 文件，包含两部分：YAML frontmatter（在 `---` 标记之间）告诉 Claude 何时使用该 skill，以及包含 Claude 在调用该 skill 时遵循的说明的 markdown 内容。
  - 目录名称变成你输入的命令，`description` 帮助 Claude 决定何时自动加载该 skill。

  ```markdown
  ---
  description: Summarizes uncommitted changes and flags anything risky. Use when the user asks what changed, wants a commit message, or asks to review their diff.
  ---
  
  ## Current changes
  
  !`git diff HEAD`
  
  ## Instructions
  
  Summarize the changes above in two or three bullet points, then list any risks you notice such as missing error handling, hardcoded values, or tests that need updating. If the diff is empty, say there are no uncommitted changes.
  ```

- 测试skill

  - **让 Claude 自动调用它**，通过询问与描述匹配的内容：

    ```
    What did I change?
    ```

  - **或直接使用 skill 名称调用它**：

    ```
    /summarize-changes
    ```

    无论哪种方式，Claude 都应该用你的编辑的简短摘要和风险列表来响应。

## 4.6 commands

命令和skill现在是同样的一套机制。new command通常应该是skill而不是command；但命令仍然得到支持

## 4.7 agents

具有各自上下文窗口的专业子代理

when it loads：当你或Claude调用它时，它在自己的上下文窗口中运行

在agents文件夹下，每个markdown文件定义一个subagent，它拥有它自己的系统提示、工具访问权限，并可以选择拥有自己的模型。subagent在全新的上下文窗口中运行，保持主对话的整洁。适用于并行工作或隔离任务。

在命令行中输入一个 **`@`**并且 pick 一个agent ，来直接委派agent

## 4.8 workflows

每一个.js文件都是一个动态的工作流程：这些脚本会在运行时被执行，以启动并协调多个子代理节点的运作。这些工作流程是由Claude编写的，并保存在此处，而不是从头开始编写的。

# 5 [skills](https://code.claude.com/docs/zh-CN/skills)

## 5.1 配置skill

Skills 通过 `SKILL.md` 顶部的 YAML frontmatter 和随后的 markdown 内容进行配置。

### 5.1.1 Frontmatter(前言)

除了 markdown 内容外，你可以使用 `SKILL.md` 文件顶部 `---` 标记之间的 YAML frontmatter 字段来配置 skill 行为。

```markdown
---
name: my-skill
description: What this skill does
disable-model-invocation: true
allowed-tools: Read Grep
---

Your skill instructions here...
```

| 字段                       | 必需 | 描述                                                         |
| :------------------------- | :--- | :----------------------------------------------------------- |
| `name`                     | 否   | Skill 列表中显示的显示名称。默认为目录名称。请参阅[Skill 如何获得其命令名称](https://code.claude.com/docs/zh-CN/skills#how-a-skill-gets-its-command-name)以了解这与你输入的名称在 `/` 后的调用方式有何不同。 |
| `description`              | 推荐 | Skill 的功能以及何时使用它。Claude 使用它来决定何时应用该 skill。如果省略，使用 markdown 内容的第一段。将关键用例放在前面：组合的 `description` 和 `when_to_use` 文本在 skill 列表中被截断为 1,536 个字符以减少上下文使用。 |
| `when_to_use`              | 否   | 关于 Claude 何时应该调用该 skill 的额外上下文，例如触发短语或示例请求。附加到 skill 列表中的 `description`，并计入 1,536 个字符的上限。 |
| `argument-hint`            | 否   | 自动完成期间显示的提示，指示预期的参数。示例：`[issue-number]` 或 `[filename] [format]`。 |
| `arguments`                | 否   | 用于 skill 内容中[`$name` 替换](https://code.claude.com/docs/zh-CN/skills#available-string-substitutions)的命名位置参数。接受空格分隔的字符串或 YAML 列表。名称按顺序映射到参数位置。 |
| `disable-model-invocation` | 否   | 设置为 `true` 以防止 Claude 自动加载此 skill。用于你想使用 `/name` 手动触发的工作流。也防止该 skill 被[预加载到 subagents](https://code.claude.com/docs/zh-CN/sub-agents#preload-skills-into-subagents) 中。从 v2.1.196 开始，也防止该 skill 在[计划任务](https://code.claude.com/docs/zh-CN/scheduled-tasks)使用该 skill 作为其提示时运行。默认值：`false`。 |
| `user-invocable`           | 否   | 设置为 `false` 以从 `/` 菜单中隐藏。用于用户不应直接调用的背景知识。默认值：`true`。 |
| `allowed-tools`            | 否   | 当此 skill 处于活动状态时，Claude 可以使用而无需请求权限的工具。接受空格分隔的字符串或 YAML 列表。 |
| `disallowed-tools`         | 否   | 当此 skill 处于活动状态时从 Claude 的可用工具池中移除的工具。用于不应该调用某些工具的自主 skills，例如用于后台循环的 `AskUserQuestion`。接受空格分隔的字符串或 YAML 列表。当你发送下一条消息时，限制会清除。 |
| `model`                    | 否   | 当此 skill 处于活动状态时要使用的模型。覆盖适用于当前轮的其余部分，不保存到设置；会话模型在你的下一个提示时恢复。接受与 [`/model`](https://code.claude.com/docs/zh-CN/model-config) 相同的值，或 `inherit` 以保持活动模型。被你的组织的 [`availableModels`](https://code.claude.com/docs/zh-CN/model-config#restrict-model-selection) 允许列表排除的值不会被使用，会话保持其当前模型。 |
| `effort`                   | 否   | 当此 skill 处于活动状态时的[工作量级别](https://code.claude.com/docs/zh-CN/model-config#adjust-effort-level)。覆盖会话工作量级别。默认值：继承自会话。选项：`low`、`medium`、`high`、`xhigh`、`max`；可用级别取决于模型。 |
| `context`                  | 否   | 设置为 `fork` 以在分叉的 subagent 上下文中运行。             |
| `agent`                    | 否   | 当设置 `context: fork` 时要使用的 subagent 类型。            |
| `hooks`                    | 否   | 限定于此 skill 生命周期的 hooks。有关配置格式，请参阅 [Skills 和代理中的 Hooks](https://code.claude.com/docs/zh-CN/hooks#hooks-in-skills-and-agents)。 |
| `paths`                    | 否   | Glob 模式，限制何时激活此 skill。接受逗号分隔的字符串或 YAML 列表。设置后，Claude 仅在处理与模式匹配的文件时自动加载该 skill。使用与[路径特定规则](https://code.claude.com/docs/zh-CN/memory#path-specific-rules)相同的格式。 |
| `shell`                    | 否   | 用于此 skill 中 `!`command`` 和 ````!` 块的 shell。接受 `bash`（默认）或 `powershell`。设置 `powershell` 在 Windows 上通过 PowerShell 运行内联 shell 命令。需要 `CLAUDE_CODE_USE_POWERSHELL_TOOL=1`。 |

#### [skill-name](https://code.claude.com/docs/zh-CN/skills#how-a-skill-gets-its-command-name)

| Skill 位置                                                   | 命令名称来源                                      | 示例                                                         |
| :----------------------------------------------------------- | :------------------------------------------------ | :----------------------------------------------------------- |
| `~/.claude/skills/` 或 `.claude/skills/` 下的 Skill 目录     | 目录名称                                          | `.claude/skills/deploy-staging/SKILL.md` → `/deploy-staging` |
| [嵌套](https://code.claude.com/docs/zh-CN/skills#where-skills-live) `.claude/skills/` 目录，当名称与另一个 skill 冲突时 | 相对于工作目录的子目录路径，然后是 skill 目录名称 | `apps/web/.claude/skills/deploy/SKILL.md` → `/apps/web:deploy` |
| `.claude/commands/` 下的文件                                 | 文件名称（不含扩展名）                            | `.claude/commands/deploy.md` → `/deploy`                     |
| 插件 `skills/` 子目录                                        | 目录名称，由插件命名空间                          | `my-plugin/skills/review/SKILL.md` → `/my-plugin:review`     |
| 插件根 `SKILL.md`                                            | Frontmatter `name`，以插件目录名称作为后备        | `my-plugin/SKILL.md` 带有 `name: review` → `/my-plugin:review`。请参阅[路径行为规则](https://code.claude.com/docs/zh-CN/plugins-reference#path-behavior-rules) |

### 5.1.2 内容(正文)

Skill 文件可以包含任何说明，skill内容类型包含：

- **参考内容**：为claude应用于你当前工作的知识

  - 知识包括：约定、模式、风格指南、领域知识

  - 此内容内联运行，以便 Claude 可以将其与你的对话上下文一起使用

  - ```markdown
    ---
    name: api-conventions
    description: API design patterns for this codebase
    ---
    
    When writing API endpoints:
    - Use RESTful naming conventions
    - Return consistent error formats
    - Include request validation
    ```

- **任务内容**：为 Claude 提供特定操作的分步说明，如部署、提交或代码生成。

  - 通常是你想使用 `/skill-name` 直接调用的操作，而不是让 Claude 决定何时运行它们。

  - 可以添加 `disable-model-invocation: true` 以防止 Claude 自动触发它。

  - ```markdown
    ---
    name: deploy
    description: Deploy the application to production
    context: fork
    disable-model-invocation: true
    ---
    
    Deploy the application:
    1. Run the test suite
    2. Build the application
    3. Push to the deployment target
    ```

### 5.1.3 支持文件

Skills 可以在其目录中包含多个文件。这使 `SKILL.md` 专注于要点，同时让 Claude 仅在需要时访问详细的参考资料。大型参考文档、API 规范或示例集合不需要在每次 skill 运行时加载到上下文中。

```markdown
my-skill/
├── SKILL.md (required - overview and navigation)
├── reference.md (detailed API docs - loaded when needed)
├── examples.md (usage examples - loaded when needed)
└── scripts/
    └── helper.py (utility script - executed, not loaded)
```

从 `SKILL.md` 中引用支持文件，以便 Claude 知道每个文件包含什么以及何时加载它

```markdown
## Additional resources

- For complete API details, see [reference.md](reference.md)
- For usage examples, see [examples.md](examples.md)
```

### 5.1.4 内容占位符

Skills 支持 skill 内容中动态值的字符串替换

| 变量                    | 描述                                                         |
| :---------------------- | :----------------------------------------------------------- |
| `$ARGUMENTS`            | 调用 skill 时传递的所有参数。如果内容中不存在 `$ARGUMENTS`，参数将作为 `ARGUMENTS: <value>` 追加。 |
| `$ARGUMENTS[N]`         | 按 0 基索引访问特定参数，如 `$ARGUMENTS[0]` 表示第一个参数。 |
| `$N`                    | `$ARGUMENTS[N]` 的简写，如 `$0` 表示第一个参数或 `$1` 表示第二个参数。 |
| `$name`                 | 在 [`arguments`](https://code.claude.com/docs/zh-CN/skills#frontmatter-reference) frontmatter 列表中声明的命名参数。名称按顺序映射到位置，因此使用 `arguments: [issue, branch]` 时，占位符 `$issue` 扩展为第一个参数，`$branch` 扩展为第二个参数。 |
| `${CLAUDE_SESSION_ID}`  | 当前会话 ID。适用于日志记录、创建会话特定文件或将 skill 输出与会话关联。 |
| `${CLAUDE_EFFORT}`      | 当前工作量级别：`low`、`medium`、`high`、`xhigh` 或 `max`。Ultracode 不是一个不同的级别，报告为 `xhigh`。使用此来根据活动工作量设置调整 skill 说明。 |
| `${CLAUDE_SKILL_DIR}`   | 包含 skill 的 `SKILL.md` 文件的目录。对于插件 skills，这是插件内 skill 的子目录，而不是插件根目录。在 bash 注入命令中使用它来引用与 skill 捆绑的脚本或文件，无论当前工作目录如何。 |
| `${CLAUDE_PROJECT_DIR}` | 项目根目录。这是与 [hooks](https://code.claude.com/docs/zh-CN/hooks#reference-scripts-by-path) 和 MCP 服务器相同的路径，作为 `CLAUDE_PROJECT_DIR` 接收。使用此来引用项目本地脚本或文件，例如 `${CLAUDE_PROJECT_DIR}/.claude/hooks/helper.sh`，独立于 skill 的安装位置。 |

### 5.1.5 内容的生命周期

当你或 Claude 调用一个 skill 时，呈现的 `SKILL.md` 内容作为单个消息进入对话，并在会话的其余部分保持在那里。

当 Claude 重新调用一个 skill 且其呈现的内容与已在上下文中的副本相同时，Claude Code 添加一个简短的说明，表示该 skill 已加载，而不是内容的第二份副本。当呈现的内容不同时，因为参数改变或[动态上下文](https://code.claude.com/docs/zh-CN/skills#inject-dynamic-context)命令产生了新输出，Claude Code 会再次附加完整内容。

## 5.2 动态上下文注入

在上面第二步SKILL.md的内容中，包含一个`!git diff HEAD`

`!git diff HEAD` 这一行使用[动态上下文注入](https://code.claude.com/docs/zh-CN/skills#inject-dynamic-context)：Claude Code 运行该命令，并在 Claude 看到 skill 内容之前将该行替换为其输出，因此说明会随着当前 diff 已内联而到达。

### 内联动态上下文

``!<command>`` 语法在将 skill 内容发送给 Claude 之前运行 shell 命令。命令输出替换占位符，因此 Claude 接收实际数据，而不是命令本身。

内联形式仅在 `!` 出现在行首或紧跟在空白之后时被识别。如果 `!` 跟在另一个字符之后，如 `KEY=!`cmd``，占位符将保留为字面文本，命令不会运行。

### 块动态上下文

对于多行命令，使用以 ````!` 开头的围栏代码块而不是内联形式

````markdown
## Environment
```!
node --version
npm --version
git status --short
```
````



## 5.3 skill 参数

`/skill-name` 是**可以携带参数**的，而且用法非常灵活。这些参数可以在技能内部通过多种方式被使用，让技能从一个固定模板变成一个能响应你具体需求的动态工具。这个也是动态上下文注入的一种方式。

### 如何传递

```bash
/code-review main.py
```

这里 `main.py` 就是传递给 `/code-review` 的参数。

从 v2.1.199 开始，你甚至可以在**一条消息的开头链接最多 6 个技能**，并将尾部文本作为参数**同时**传递给每个技能。

```bash
/code-review /fix-issue 123
```

这个命令会同时加载 `code-review` 和 `fix-issue` 两个技能，并将 `123` 作为参数传递给它们两个

### 参数如何生效

在md文件中增加占位符，预处理占位符，然后交给claude code阅读。

| 占位符                | 说明                                                         | 示例                                                         |
| :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `$ARGUMENTS`          | 代表调用时传递的**全部参数**。如果技能内容中没有这个占位符，参数会以 `ARGUMENTS: <value>` 的形式附加在末尾。 | 调用 `/greet Hello world`，`$ARGUMENTS` 的值就是 `Hello world`。 |
| `$0`, `$1`, ..., `$N` | 用于访问按**位置**排序的单个参数（从0开始）。                | 调用 `/deploy main production`, `$0` 是 `main`, `$1` 是 `production`。 |
| `$ARGUMENTS[0]`       | 功能同 `$0`，是按索引访问参数的一种更显式的写法。            | 调用 `/deploy main`, `$ARGUMENTS[0]` 的值是 `main`。         |
| `$name`               | 这是**命名参数**，需要在技能的 `arguments` Frontmatter 字段中预先声明。声明后，参数会根据位置映射到对应的名称。 | 声明 `arguments: [issue, branch]`，调用 `/fix-issue 123 main` 时，`$issue` 是 `123`，`$branch` 是 `main`。 |

### 参数提示更友好

在 `SKILL.md` 的 Frontmatter（文件开头的 YAML 元数据区）中添加 `argument-hint` 字段，方便用户在输入 `/skill-name` 后看到参数提示。

```markdown
---
name: create-component
description: 创建新的 React 组件
arguments: [name, type]
argument-hint: <component-name> [functional|class]
---
```

用户在输入 `/create-component` 后看到

```
/create-component <component-name> [functional|class]
```

## 5.4 变更检测与发现skill

### 变更检测

Claude Code 监视 skill 目录的文件**变更**

- 在 `~/.claude/skills/`、项目 `.claude/skills/` 或 `--add-dir` 目录内的 `.claude/skills/` 中添加、编辑或删除 skill 会在当前会话中生效，无需重新启动。

**创建**在会话启动时不存在的顶级 skills 目录

- 需要重新启动 Claude Code，以便可以监视新目录。

### 发现

项目 skills 从你的起始目录中的 `.claude/skills/` 以及从起始目录到仓库根目录的每个父目录中加载，因此在子目录中启动 Claude 仍然会拾取在根目录定义的 skills。

这个发现也需要重新启动claude code

#### add-dir

`--add-dir` 标志和 `/add-dir` 命令[授予文件访问权限](https://code.claude.com/docs/zh-CN/permissions#additional-directories-grant-file-access-not-configuration)而不是配置发现，但 skills 是一个例外：添加目录中的 `.claude/skills/` 会自动加载。



## 5.5 subagent 中运行skill

当你想让 skill 在隔离中运行时，在你的 frontmatter 中添加 `context: fork`。skill 内容变成驱动 subagent 的提示。它将无法访问你的对话历史。

`SKILL.md`中要有**任务内容**，如果仅包含参考内容，那么是没有意义的。subagent 仅收到指南但没有可操作的内容，那么返回就没有有意义的输出。

使用 `context: fork`，你在你的 skill 中编写任务并选择一个代理类型来执行它。内置的 Explore 和 Plan 代理[跳过 CLAUDE.md 和 git 状态](https://code.claude.com/docs/zh-CN/sub-agents#what-loads-at-startup)以保持其上下文较小，因此使用 `agent: Explore` 的分叉 skill 仅看到 SKILL.md 内容和代理自己的系统提示。

```markdown
---
name: deep-research
description: Research a topic thoroughly
context: fork
agent: Explore
---

Research $ARGUMENTS thoroughly:

1. Find relevant files using Glob and Grep
2. Read and analyze the code
3. Summarize findings with specific file references
```

当此 skill 运行时：

1. 创建一个新的隔离上下文
2. Subagent 接收 skill 内容作为其提示（“Research $ARGUMENTS thoroughly…”）
3. `agent` 字段确定执行环境（模型、工具和权限）
4. 结果被总结并返回到你的主对话

`agent` 字段指定要使用的 subagent 配置。选项包括内置代理（`Explore`、`Plan`、`general-purpose`）或来自 `.claude/agents/` 的任何自定义 subagent。如果省略，使用 `general-purpose`。

# 6 [agents](https://code.claude.com/docs/zh-CN/sub-agents)

在Claude Code中创建和使用专门的AI agent，用于特定任务的工作流和改进的上下文管理。

Subagents 是处理特定类型任务的专门 AI 助手

Subagents 帮助您：

- **保留上下文**，通过将探索和实现保持在主对话之外
- **强制执行约束**，通过限制 subagent 可以使用的工具
- **跨项目重用配置**，使用用户级 subagents
- **专门化行为**，为特定领域使用专注的系统提示
- **控制成本**，通过将任务路由到更快、更便宜的模型（如 Haiku）



## 6.1 内置(built-in) subagents

Claude Code 包括内置 subagents，Claude 在适当时自动使用。每个都继承父对话的权限，并有额外的工具限制。

内置的subagent

- Explore
- Plan
- General-purpose

Explore 和 Plan 会跳过您的 CLAUDE.md 文件和父会话的 git 状态，以保持研究快速且成本低廉。所有其他内置和[自定义 subagent](https://code.claude.com/docs/zh-CN/sub-agents#configure-subagents) 都会加载两者。请参阅[启动时加载的内容](https://code.claude.com/docs/zh-CN/sub-agents#what-loads-at-startup)。

内置 subagents 在交互式会话中默认被注册。

### Explore

一个快速的、只读的代理，针对搜索和分析代码库进行了优化

- **Model**: 从主对话继承
- **Tools**: 只读工具；拒绝访问 Write 和 Edit
- **Purpose**: 文件发现、代码搜索、代码库探索

当 Claude 需要搜索或理解代码库而不进行更改时，它会委托给 Explore。这样可以将探索结果保持在主对话上下文之外。	

### Plan

一个研究代理，在 [plan mode](https://code.claude.com/docs/zh-CN/permission-modes#analyze-before-you-edit-with-plan-mode) 期间使用，以在呈现计划之前收集上下文。

- **Model**: 从主对话继承
- **Tools**: 只读工具；拒绝访问 Write 和 Edit
- **Purpose**: 用于规划的代码库研究

当您处于 plan mode 并且 Claude 需要理解您的代码库时，它会将研究委托给 Plan subagent，以便探索输出保持在单独的上下文窗口中

### General-purpose

一个能够处理复杂、多步骤任务的代理，需要探索和操作。

- **Model**: 从主对话继承
- **Tools**: 所有工具
- **Purpose**: 复杂研究、多步骤操作、代码修改

当任务需要探索和修改、复杂推理来解释结果或多个依赖步骤时，Claude 会委托给 general-purpose。

## 阻止特定agent

settings.json中

```
{
  "permissions": {
    "deny": ["Agent(Explore)", "Agent(my-custom-agent)"]
  }
}
```

## 6.2 创建subagent

在agents目录下，Subagents 是带有 YAML frontmatter 的 Markdown 文件。

示例创建一个code-improver.md

```markdown
---
name: code-improver
description: Scans files and suggests improvements for readability, performance, and best practices. Use after writing or modifying code.
tools: Read, Grep, Glob
model: sonnet
---

You are a code improvement specialist. For each issue you find, explain
the problem, show the current code, and provide an improved version.
```

如果 Claude 找不到新的 subagent，请重新启动 Claude Code 并重试。这仅在会话开始前 `~/.claude/agents/` 不存在时发生，因为运行中的会话不会检测到新创建的 `agents` 目录。

## 6.3 配置subagents

### frontmatter

以下字段可以在 YAML frontmatter 中使用。只有 name 和 description 是必需的。

| Field             | 必需 | Description                                                  |
| :---------------- | :--- | :----------------------------------------------------------- |
| `name`            | 是   | 使用小写字母和连字符的唯一标识符。[Hooks](https://code.claude.com/docs/zh-CN/hooks#subagentstart) 将此值作为 `agent_type` 接收。文件名不必匹配 |
| `description`     | 是   | Claude 何时应该委托给此 subagent                             |
| `tools`           | 否   | [Tools](https://code.claude.com/docs/zh-CN/sub-agents#available-tools) subagent 可以使用。如果省略，继承所有工具。要将 Skills 预加载到上下文中，请使用 `skills` 字段而不是在此处列出 `tools` |
| `disallowedTools` | 否   | 要拒绝的工具，从继承或指定的列表中删除                       |
| `model`           | 否   | [Model](https://code.claude.com/docs/zh-CN/sub-agents#choose-a-model) 使用：`sonnet`、`opus`、`haiku`、`fable`、完整模型 ID（例如，`claude-opus-4-8`）或 `inherit`。默认为 `inherit` |
| `permissionMode`  | 否   | [Permission mode](https://code.claude.com/docs/zh-CN/sub-agents#permission-modes)：`default`、`acceptEdits`、`auto`、`dontAsk`、`bypassPermissions`、`plan` 或 `manual` 作为 `default` 的别名。`manual` 别名需要 Claude Code v2.1.200 或更高版本。对于 [plugin subagents](https://code.claude.com/docs/zh-CN/sub-agents#choose-the-subagent-scope) 被忽略 |
| `maxTurns`        | 否   | subagent 停止前的最大代理轮数                                |
| `skills`          | 否   | [Skills](https://code.claude.com/docs/zh-CN/skills) 在启动时加载到 subagent 的上下文中。注入完整的技能内容，而不仅仅是描述。Subagents 仍然可以通过 Skill 工具调用未列出的项目、用户和 plugin 技能 |
| `mcpServers`      | 否   | [MCP servers](https://code.claude.com/docs/zh-CN/mcp) 对此 subagent 可用。每个条目要么是引用已配置服务器的服务器名称（例如，`"slack"`），要么是内联定义，其中服务器名称为键，完整的 [MCP server config](https://code.claude.com/docs/zh-CN/mcp#installing-mcp-servers) 为值。对于 [plugin subagents](https://code.claude.com/docs/zh-CN/sub-agents#choose-the-subagent-scope) 被忽略 |
| `hooks`           | 否   | [Lifecycle hooks](https://code.claude.com/docs/zh-CN/sub-agents#define-hooks-for-subagents) 限定于此 subagent。对于 [plugin subagents](https://code.claude.com/docs/zh-CN/sub-agents#choose-the-subagent-scope) 被忽略 |
| `memory`          | 否   | [Persistent memory scope](https://code.claude.com/docs/zh-CN/sub-agents#enable-persistent-memory)：`user`、`project` 或 `local`。启用跨会话学习 |
| `background`      | 否   | 设置为 `true` 以始终将此 subagent 作为 [background task](https://code.claude.com/docs/zh-CN/sub-agents#run-subagents-in-foreground-or-background) 运行，即使 Claude 需要其结果。未设置时，Claude 选择，从 v2.1.198 开始，它默认在后台运行 subagents |
| `effort`          | 否   | 此 subagent 活跃时的努力级别。覆盖会话努力级别。默认：从会话继承。选项：`low`、`medium`、`high`、`xhigh`、`max`；可用级别取决于模型 |
| `isolation`       | 否   | 设置为 `worktree` 以在临时 [git worktree](https://code.claude.com/docs/zh-CN/worktrees) 中运行 subagent，为其提供存储库的隔离副本，默认从您的 [default branch](https://code.claude.com/docs/zh-CN/worktrees#choose-the-base-branch) 分支，而不是父会话的 `HEAD`。如果 subagent 不进行任何更改，worktree 会自动清理 |
| `color`           | 否   | Subagent 在任务列表和转录中的显示颜色。接受 `red`、`blue`、`green`、`yellow`、`purple`、`orange`、`pink` 或 `cyan` |
| `initialPrompt`   | 否   | 当此代理作为主会话代理运行时（通过 `--agent` 或 `agent` 设置），自动提交为第一个用户轮次。[Commands](https://code.claude.com/docs/zh-CN/commands) 和 [skills](https://code.claude.com/docs/zh-CN/skills) 被处理。前置于任何用户提供的提示 |

### 控制subagent能力

Subagents 默认继承主对话中可用的 [internal tools](https://code.claude.com/docs/zh-CN/tools-reference) 和 MCP 工具。

- `tools` 字段中列出也不可用于 subagents
  - `AskUserQuestion`
  - `EnterPlanMode`
  - `ExitPlanMode`，除非 subagent 的 [`permissionMode`](https://code.claude.com/docs/zh-CN/sub-agents#permission-modes) 是 `plan`
  - `ScheduleWakeup`
  - `WaitForMcpServers`

- 工具限制：使用 `tools` 字段（允许列表）或 `disallowedTools` 字段（拒绝列表）

- subagent限制调用subagent：当subagent作为主线程运行时（成为主agent，使用 `claude --agent`），它可以使用 Agent 工具生成 subagents。要限制它可以生成的 subagent 类型，在 `tools` 字段中使用 `Agent(agent_type)` 语法。

  ```markdown
  ---
  name: coordinator
  description: Coordinates work across specialized agents
  tools: Agent(worker, researcher), Read, Bash
  ---
  
  要允许生成任何 subagent 而不受限制，使用不带括号的 Agent
  tools: Agent, Read, Bash
  
  如果 Agent 完全从 tools 列表中省略，代理无法生成任何 subagents。
  ```

- 限制mcp使用范围

  ```markdown
  ---
  name: browser-tester
  description: Tests features in a real browser using Playwright
  mcpServers:
    # Inline definition: scoped to this subagent only
    - playwright:
        type: stdio
        command: npx
        args: ["-y", "@playwright/mcp@latest"]
    # Reference by name: reuses an already-configured server
    - github
  ---
  
  Use the Playwright tools to navigate, screenshot, and interact with pages.
  ```

- 权限模式：`permissionMode` 字段控制 subagent 如何处理权限提示。Subagents 从主对话继承权限上下文，并可以覆盖模式

  | Mode                | Behavior                                                     |
  | :------------------ | :----------------------------------------------------------- |
  | `default`           | 标准权限检查，带有提示                                       |
  | `acceptEdits`       | 自动接受文件编辑和工作目录或 `additionalDirectories` 中路径的常见文件系统命令 |
  | `auto`              | [Auto mode](https://code.claude.com/docs/zh-CN/permission-modes#eliminate-prompts-with-auto-mode)：后台分类器审查命令和受保护目录的写入 |
  | `dontAsk`           | 自动拒绝权限提示。显式允许的工具仍然工作；`AskUserQuestion`、connector 工具 [您的组织设置为 `ask`](https://code.claude.com/docs/zh-CN/mcp#organization-controls-on-connector-tools) 和标记为 [`requiresUserInteraction`](https://code.claude.com/docs/zh-CN/mcp#require-approval-for-a-specific-tool) 的 MCP 工具被拒绝，即使您已允许它们 |
  | `bypassPermissions` | 跳过权限提示                                                 |
  | `plan`              | Plan mode（只读探索）                                        |

- 预加载skill到subagent：

  - 预加载使用 `skills` 字段在启动时将技能内容注入到 subagent 的上下文中。这为 subagent 提供领域知识，而无需在执行期间发现和加载技能。

  - 每个列出的技能的完整内容被注入到 subagent 的上下文中

  - 此字段控制哪些技能被预加载，而不是 subagent 可以访问哪些技能：没有它，subagent 仍然可以在执行期间通过 Skill 工具发现和调用项目、用户和 plugin 技能。要防止 subagent 完全调用技能，请从 [`tools`](https://code.claude.com/docs/zh-CN/sub-agents#available-tools) 列表中省略 `Skill` 或将其添加到 `disallowedTools`。

  - ```markdown
    ---
    name: api-developer
    description: Implement API endpoints following team conventions
    skills:
      - api-conventions
      - error-handling-patterns
    ---
    
    Implement API endpoints. Follow the conventions and patterns from the preloaded skills.
    ```

- 使用持久记忆memory：`memory` 字段为 subagent 提供一个在对话中幸存的持久目录。Subagent 使用此目录随时间积累知识。

  ```markdown
  ---
  name: code-reviewer
  description: Reviews code for quality and best practices
  memory: user
  ---
  
  You are a code reviewer. As you review code, update your agent memory with
  patterns, conventions, and recurring issues you discover.
  ```

  | Scope     | Location                                      | 使用时机                                          |
  | :-------- | :-------------------------------------------- | :------------------------------------------------ |
  | `user`    | `~/.claude/agent-memory/<name-of-agent>/`     | subagent 应该在所有项目中记住学习                 |
  | `project` | `.claude/agent-memory/<name-of-agent>/`       | subagent 的知识是特定于项目的并可通过版本控制共享 |
  | `local`   | `.claude/agent-memory-local/<name-of-agent>/` | subagent 的知识是特定于项目的但不应检入版本控制   |

- 禁用特定的agent

  - 在settings.json，防止 Claude 使用特定 subagents。

  ```json
  {
    "permissions": {
      "deny": ["Agent(Explore)", "Agent(my-custom-agent)"]
    }
  }
  ```

- 为subagent配置hooks





## 6.4 使用agent

### 6.4.1 自动委托

Claude 根据您请求中的任务描述、subagent 配置中的 `description` 字段和当前上下文自动委托任务。

鼓励：subagent的description字段中包含“主动地使用”等词语

### 6.4.2 显式调用subagent

当自动委托不够时，您可以自己请求 subagent。

- **自然语言**：在提示中命名 subagent；Claude 决定是否委托

- **@-mention**：保证 subagent 为一个任务运行

  - 输入 `@` 并从类型提前中选择 subagent，就像您 @-mention 文件一样。这确保特定 subagent 运行，而不是将选择留给 Claude

  ```
  @"code-reviewer (agent)" look at the auth changes
  ```

- **会话范围**：整个会话使用该 subagent 的系统提示、工具限制和模型，通过 `--agent` 标志或 `agent` 设置

  会话参数

  ```
  claude --agent security-reviewer
  ```

  .claude/settings.json

  ```json
  {
    "agent": "code-reviewer"
  }
  ```

  

### 6.4.3 在前台或后台调用subagent

Subagents 可以在前台或后台运行

- **前台 subagents** 阻塞主对话直到完成。权限提示会在出现时传递给您。
- **后台 subagents** 在您继续工作时并发运行。从 v2.1.186 开始，当后台 subagent 到达需要权限的工具调用时，提示会在您的主会话中显示，并命名正在请求的 subagent。批准以让 subagent 继续，或按 Esc 拒绝该单个工具调用而不停止 subagent。在 v2.1.186 之前，后台 subagents 自动拒绝任何会提示的工具调用。

从 v2.1.198 开始，subagents 默认在后台运行。Claude 在需要结果才能继续时在前台运行 subagent。

您也可以自己控制这个：

- 要求 Claude 在后台或前台运行任务
- 按 **Ctrl+B** 将运行中的任务放在后台

完成的后台 subagent 在 [`/tasks`](https://code.claude.com/docs/zh-CN/commands) 中保持列出，标记为完成并排序在运行工作下方，直到会话清理其任务列表。

### 6.4.4 常见使用模式

1. 隔离高容量操作
   - subagents 最有效的用途之一是隔离产生大量输出的操作。
   - 运行测试、获取文档或处理日志文件可能会消耗大量上下文。通过将这些委托给 subagent，详细输出保留在 subagent 的上下文中，而只有相关摘要返回到您的主对话。
2. 运行并行研究
   - 对于独立的调查，生成多个 subagents 以同时工作
   - 每个 subagent 独立探索其区域，然后 Claude 综合这些发现。当研究路径彼此不依赖时，这效果最好。
   - 当 subagents 完成时，它们的结果返回到您的主对话。运行许多 subagents，每个都返回详细结果，可能会消耗大量上下文。
3. chain subagent
   - 对于多步骤工作流，要求 Claude 按顺序使用 subagents。每个 subagent 完成其任务并将结果返回给 Claude，然后将相关上下文传递给下一个 subagent。

### subagent和主对话的选择

在以下情况下使用 **主对话**：

- 任务需要频繁的来回或迭代细化
- 多个阶段共享重要上下文，例如规划、实现和测试
- 您正在进行快速、有针对性的更改
- 延迟很重要。Subagents 从头开始，可能需要时间来收集上下文

在以下情况下使用 **subagents**：

- 任务产生 不需要在主上下文中的详细输出
- 您想强制执行特定的工具限制或权限
- 工作是自包含的，可以返回摘要



### 6.4.5 嵌套的subagent

subagent 可以生成自己的 subagents。当委托的任务本身分裂成并行子任务时使用这个，例如审查者 subagent 为每个发现分派一个验证者，所以中间输出永远不会到达您的主对话。只有顶级 subagent 的摘要返回给您。



### 6.4.6 分叉当前对话

分叉是一个 subagent，它继承到目前为止的整个对话，而不是从头开始。

这消除了 subagents 通常提供的输入隔离：分叉看到与主会话相同的系统提示、工具、模型和消息历史，因此您可以将其交给一个辅助任务而无需重新解释情况。

分叉自己的工具调用仍然保持在您的对话之外，只有其最终结果返回，因此您的主 context window 保持干净。

当命名 subagent 需要太多背景才能有用时，或当您想从相同的起点并行尝试多种方法时，使用分叉。

`CLAUDE_CODE_FORK_SUBAGENT`默认为0，设置为1以让claude生成分叉subagent，显式 [`/fork`](https://code.claude.com/docs/zh-CN/commands) 命令无需此变量即可工作。

每个subagent生成都在background中运行，无论是分叉还是命名的subagent。

```bash
/fork draft unit tests for the parser changes so far
```

分叉出现在提示下方的面板中，并在您继续工作时在后台运行。完成后，其结果作为消息到达您的主对话。下一部分涵盖了在分叉运行时观察和引导它们的面板控制。

#### 观察和引导运行中的分叉

运行中的分叉出现在提示输入下方的面板中，主会话有一行，每个分叉有一行。使用这些键与面板交互

| Key       | Action                               |
| :-------- | :----------------------------------- |
| `↑` / `↓` | 在行之间移动                         |
| `Enter`   | 打开所选分叉的转录并向其发送后续消息 |
| `x`       | 关闭完成的分叉或停止运行中的分叉     |
| `Esc`     | 将焦点返回到提示输入                 |

#### 分叉与命名subagent的区别

| 分叉           | 命名 subagent        |                                                              |
| :------------- | :------------------- | ------------------------------------------------------------ |
| 上下文         | 完整的对话历史       | 新鲜上下文，带有您传递的提示                                 |
| 系统提示和工具 | 与主会话相同         | 来自 subagent 的 [definition file](https://code.claude.com/docs/zh-CN/sub-agents#write-subagent-files) |
| 模型           | 与主会话相同         | 来自 subagent 的 `model` 字段                                |
| 权限           | 提示在您的终端中出现 | [提示在后台运行时在您的主会话中出现](https://code.claude.com/docs/zh-CN/sub-agents#run-subagents-in-foreground-or-background) |
| Prompt cache   | 与主会话共享         | 单独的缓存                                                   |

### 6.4.7 管理subagent上下文

每个 subagent 都以新鲜的隔离上下文窗口开始。它看不到您的对话历史、您已经调用的技能或 Claude 已经读取的文件。Claude 编写一条委托消息来总结任务，subagent 从那里开始工作。（除fork外的subagent）

#### 启动时的上下文

非 fork subagent 的初始上下文包含：

- **系统提示**：代理自己的提示加上 Claude Code 附加的环境详情，而不是完整的 Claude Code 系统提示。自定义 subagents 在 [markdown 正文](https://code.claude.com/docs/zh-CN/sub-agents#write-subagent-files) 或 `prompt` 字段中定义它们。内置代理有预定义的提示。
- **任务消息**：Claude 在移交工作时编写的委托提示。
- **CLAUDE.md 和内存**：主对话加载的 [内存层次结构](https://code.claude.com/docs/zh-CN/memory#how-claude-md-files-load) 的每个级别，包括 `~/.claude/CLAUDE.md`、项目规则、`CLAUDE.local.md` 和托管策略文件。内置的 Explore 和 Plan 代理跳过这个。
- **Git 状态**：在父会话开始时拍摄的快照。当工作目录不是 Git 存储库或 [`includeGitInstructions`](https://code.claude.com/docs/zh-CN/settings#available-settings) 为 `false` 时不存在。Explore 和 Plan 无论如何都跳过它。
- **预加载的技能**：代理的 [`skills` 字段](https://code.claude.com/docs/zh-CN/sub-agents#preload-skills-into-subagents) 中命名的任何技能的完整内容。内置代理不预加载技能。
- **兄弟名单**：系统提醒，列出 `main` 和会话中的每个其他命名代理，每个都是 [`SendMessage`](https://code.claude.com/docs/zh-CN/sub-agents#resume-subagents) 的有效 `to` 值。需要 Claude Code v2.1.206 或更高版本。名单仅在 subagent 的工具包括 `SendMessage` 且至少有一个其他代理有名称时出现，无论 Claude 在生成时命名它还是它作为 [agent team](https://code.claude.com/docs/zh-CN/agent-teams) 队友运行。它是 subagent 启动时拍摄的快照，所以稍后命名的代理不会出现。

#### 恢复subagents

每个 subagent 调用都会创建一个具有新鲜上下文的新实例。要继续现有 subagent 的工作而不是重新开始，要求 Claude 恢复它。恢复的 subagents 保留其完整的对话历史，包括所有以前的工具调用、结果和推理。Subagent 从它停止的地方继续，而不是从头开始。

当 subagent 完成时，Claude 接收其代理 ID。内置的 Explore 和 Plan 代理是一次性的，不返回代理 ID，所以它们无法恢复；当您需要继续工作时，使用 `general-purpose` 或自定义 subagent。Claude 使用 `SendMessage` 工具，将代理的 ID 或名称作为 `to` 字段来恢复它。完成的 subagent 如果接收 `SendMessage`，会在后台自动恢复，无需新的 `Agent` 调用。同样适用于 Claude 用 `TaskStop` 工具停止的 subagent。

Subagent Transcript（转录）独立于主对话持久化：

- **主对话压缩**：当主对话压缩时，subagent 转录不受影响。它们存储在单独的文件中。
- **会话持久性**：Subagent 转录在其会话中持久化。您可以通过恢复相同的会话在重启 Claude Code 后 [恢复 subagent](https://code.claude.com/docs/zh-CN/sub-agents#resume-subagents)。
- **自动清理**：转录根据 `cleanupPeriodDays` 设置（默认为 30 天）进行清理。

#### 自动压缩

Subagents 支持使用与主对话相同的逻辑进行自动压缩。在env的配置中：

- `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`：设置触发自动压缩的上下文容量百分比（1-100）
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`：设置用于自动压缩计算的上下文容量（以令牌为单位）



# 7 agent teams

协调多个 Claude Code 实例作为一个团队一起工作，具有共享任务、代理间消息传递和集中管理。

与subagents不同，subagent只能向master报告，而在agent teams中，每个agent可以相互通信，而无需让master作为中间人。

agent teams是实验性功能，默认禁用，`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`可以禁用

![比较 subagent 和 agent team 架构的图表。Subagents 由主代理生成、执行工作并报告结果。Agent teams 通过共享任务列表进行协调，队友彼此直接通信。](legend/subagents-vs-agent-teams-light.png)

## 7.1 适用场景

- **研究和审查**：多个队友可以同时调查问题的不同方面，然后分享和质疑彼此的发现
- **新模块或功能**：队友可以各自拥有一个独立的部分，不会相互干扰
- **使用竞争假设进行调试**：队友并行测试不同的理论，更快地收敛到答案
- **跨层协调**：跨越前端、后端和测试的更改，每个由不同的队友负责

## 7.2 控制teams

### 7.2.1 启用agent teams

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

启用 agent teams 后，用自然语言描述你想要的任务和队友。Claude 会生成他们并根据你的提示协调工作。

```
I'm designing a CLI tool that helps developers track TODO comments across
their codebase. Spawn three teammates to explore this from different angles:
one on UX, one on technical architecture, one playing devil's advocate.

我正在设计一个命令行工具，旨在帮助开发者在代码库中跟踪待办事项评论。我会安排三位队友从不同的角度来研究这个问题：一位负责用户体验方面，一位负责技术架构设计，还有一位则扮演“反对派”角色，提出一些不同的观点。
```

从那里，Claude 会填充一个 [共享任务列表](https://code.claude.com/docs/zh-CN/interactive-mode#task-list)，为每个角度生成队友，让他们探索问题，并在完成时综合发现。

负责人的终端在提示输入下方的 agent 面板中列出队友。从该面板中：

- **向上和向下箭头**：选择一个队友
- **Enter**：打开所选队友的记录并直接向其发送消息
- **Escape**：中断所选队友的当前轮次

# 8 workflows

动态工作流是一个 JavaScript 脚本，可大规模编排subagent。

Claude 为您描述的任务编写脚本，运行时在后台执行它，同时您的会话保持响应。

## 8.1 让Claude 编写工作流

您可以通过两种方式让 Claude 为您的任务编写工作流：

- [在您的提示中请求工作流](https://code.claude.com/docs/zh-CN/workflows#ask-for-a-workflow-in-your-prompt)，使用关键字 `ultracode`，Claude 为任务编写一个。
- [让 Claude 使用 ultracode 决定](https://code.claude.com/docs/zh-CN/workflows#let-claude-decide-with-ultracode)：设置 `/effort ultracode`，Claude 为会话中的每个实质性任务规划工作流。

您也可以运行已存在的工作流命令：一个[捆绑工作流](https://code.claude.com/docs/zh-CN/workflows#bundled-workflows)如 `/deep-research`，或一个您已[保存](https://code.claude.com/docs/zh-CN/workflows#save-the-workflow-for-reuse)的。

# Reference

## [Tool Reference](https://code.claude.com/docs/en/tools-reference)

**Claude Code 可用工具的完整参考，包括权限要求和每个工具的行为。**

## [Env Reference](https://code.claude.com/docs/zh-CN/env-vars)

控制 Claude Code 行为的环境变量完整参考

# 其他内容-----------------------------------------------------------

# 1 [claude-code-router]( https://musistudio.github.io/claude-code-router/zh-CN/)

```bash
npm install -g @musistudio/claude-code-router
ccr -v
claude-code-router version: 2.0.0
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
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions -H "Authorization: Bearer ark-57ba57e6-xxxx" -H "Content-Type: application/json" -d '{
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
