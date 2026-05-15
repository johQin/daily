# CodeBuddy

# 1 绪论

## 1.1 安装

```bash
# 安装
npm install -g @tencent-ai/codebuddy-code

# 验证
codebuddy --version

# CodeBuddy Code 默认会自动保持最新状态，以确保你拥有最新的功能和安全修复。
# 关闭自动更新
export DISABLE_AUTOUPDATER=1
```

## 1.2 配置目录

CodeBuddy Code 默认将配置文件存储在以下目录：

- Windows ：`%USERPROFILE%\.codebuddy`
- macOS / Linux ：`~/.codebuddy`

配置目录的内容

```bash
~/.codebuddy/
├── settings.json      # 用户设置
├── mcp.json           # MCP 服务器配置
├── .mcp.json          # MCP 服务器配置（备选位置）
└── skills/            # 用户自定义 Skills
```

自定义配置目录的位置：

```bash
export CODEBUDDY_CONFIG_DIR="$HOME/.my-codebuddy-config"
```

## 1.3 常见工作流

### 1.3.1 子代理

使用专门的 AI 子代理来更有效地处理特定任务。

```bash
# 查看可用的子代理
/agents

# 可以使用语言，明确让子代理工作
# eg: 使用代码审查子代理检查认证模块
```

也可以为工作流创建自定义代理

也能在 `.codebuddy/agents/` 中创建项目特定的子代理

### 1.3.2 计划模式

计划模式指示 CodeBuddy 通过只读操作分析代码库来创建计划，非常适合探索代码库、规划复杂更改或安全地审查代码。

何时使用计划模式：

- **多步骤实现**：当您的功能需要编辑许多文件时
- **代码探索**：当您想在更改任何内容之前彻底研究代码库时
- **交互式开发**：当您想与 CodeBuddy 就方向进行迭代时

#### 开启计划模式

- **macOS/Linux**：按 **Shift+Tab**
- **Windows**：按 **Alt+M**

```bash
# 以计划模式启动新会话
codebuddy --permission-mode plan
```

```json
// .codebuddy/settings.json
{
  "permissions": {
    "defaultMode": "plan"
  }
}
```

### 1.3.3 处理图片

**将图片添加到对话中** 您可以使用以下任一方法：

- 将图片拖放到 CodeBuddy Code 窗口中
- 复制图片并用 ctrl+v 粘贴到 CLI 中（不要使用 cmd+v)
- 向 CodeBuddy 提供图片路径。例如："分析这张图片： /path/to/your/image.png"

### 1.3.4 引用文件和目录

使用 @ 快速包含文件或目录

### 1.3.5 深入思考

思考模式在 CodeBuddy Code 中默认禁用。您可以使用 `Tab` 按需开启思考，或使用"思考"或"深入思考"等提示词。

扩展思考对复杂任务最有价值，例如：

- 规划复杂的架构更改
- 调试复杂问题
- 为新功能创建实现计划
- 理解复杂的代码库
- 评估不同方法之间的权衡

**注意**: CodeBuddy 会在响应上方以斜体灰色文本显示其思考过程。

### 1.3.6 恢复对话

- 使用 `--continue` 快速访问最近的对话
- 使用 `--resume` 当您需要选择特定的过去对话时

### 1.3.7 git worktrees

Git worktrees 允许您将同一仓库的多个分支检出到单独的目录中。

每个 worktree 都有自己的工作目录和隔离的文件,同时共享相同的 Git 历史。

使用 Git Worktrees 运行并行 CodeBuddy Code 会话

场景：假设您需要同时处理多个任务，并在 CodeBuddy Code 实例之间完全隔离代码。

```bash
# 1.创建新的worktree
# 用新分支创建新 worktree
git worktree add ../project-feature-a -b feature-a
# 或用现有分支创建 worktree
git worktree add ../project-bugfix bugfix-123

# 2.在每个 worktree 中运行 CodeBuddy Code
# 导航到您的 worktree，在这个隔离环境中运行 CodeBuddy Code
cd ../project-feature-a
codebuddy

cd ../project-bugfix
codebuddy

# 3.管理 worktrees
# 列出所有 worktrees
git worktree list
# 完成后删除 worktree
git worktree remove ../project-feature-a
```

### 1.3.8 命令支持管道

```bash
codebuddy -p '你是一个 linter。请查看相对于 main 的更改并报告任何与拼写错误相关的问题。在一行上报告文件名和行号，在第二行上报告问题描述。不要返回任何其他文本。'

cat build-error.txt | codebuddy -p '简洁地解释此构建错误的根本原因' > output.txt

cat data.txt | codebuddy -p '总结这些数据' --output-format text > summary.txt
cat code.py | codebuddy -p '分析此代码的 bug' --output-format json > analysis.json
cat log.txt | codebuddy -p '解析此日志文件的错误' --output-format stream-json
```

### 1.3.9 [斜杠命令](https://www.codebuddy.cn/docs/cli/slash-commands)



#### 项目斜杠命令

假设您想为项目创建可重用的斜杠命令，以便所有团队成员都可以使用。

```bash
# 在项目中创建命令目录
mkdir -p .codebuddy/commands
# 为每个命令创建 Markdown 文件
echo "分析此代码的性能并建议三个具体的优化:" > .codebuddy/commands/optimize.md
# 在 CodeBuddy Code 中使用自定义命令
/optimize
```

#### 命令参数

```bash
echo '查找并修复问题 #$ARGUMENTS。按照以下步骤: 1. 理解工单中描述的问题 2. 在代码库中定位相关代码 3. 实现解决根本原因的方案 4. 添加适当的测试 5. 准备简洁的 MR 描述' > .codebuddy/commands/fix-issue.md

# 这会将提示中的 $ARGUMENTS 替换为"123"。
> /fix-issue 123

```

#### 个人斜杠命令

```bash
mkdir -p ~/.codebuddy/commands
echo "审查此代码的安全漏洞,重点关注:" > ~/.codebuddy/commands/security-review.md
```



### 1.3.10 配置

假设您需要配置 Bash 超时、创建自定义 Skill、设置 Git Hook 或调整 CodeBuddy 的各种行为，但不想花时间查找文档或手动编辑配置文件。

**最佳实践:**

- 直接告诉 CodeBuddy 你想要什么，加上"参考 CodeBuddy Code 官方文档"前缀确保获得准确信息
- 让 CodeBuddy 生成配置文件，而不是手动编写，避免语法错误
- 配置完成后让 CodeBuddy 验证是否正确，确保配置生效
- 询问配置的优先级和继承规则，理解复杂配置场景
- 只有在 CodeBuddy 无法解决时，才需要查阅更深入的文档或联系技术支持

### 1.3.11 询问CodeBuddy自身的功能

CodeBuddy 内置了对其官方文档的访问，可以回答关于自己功能、限制和最佳实践的问题。

在询问时加上"参考 CodeBuddy Code 官方文档"前缀，可以确保获得准确的官方信息。

CodeBuddy 始终可以访问最新的 CodeBuddy Code 文档，无论您使用的版本如何

提出具体问题以获得详细答案，可以询问：

- 功能特性和使用方法
- 配置选项和语法
- 最佳实践和注意事项
- 限制和已知问题
- 集成和扩展方式

## 1.4 交互模式

键盘快捷方式，vim编辑模式，后台bash命令，命令历史，权限模式



权限模式：

- 普通模式：根据权限规则询问工具使用确认
- 自动编辑模式：自动批准文件编辑操作 （Edit/Write),其他工具仍需确认
- 跳过权限模式：绕过所有权限检查。
- 计划模式：AI 将制定计划并等待批准后再执行
- 快捷键切换：Windows（Alt + M），Mac/Linux（Shift + Tab）

```bash
# 指定启动模式
codebuddy --permission-mode acceptEdits
codebuddy --permission-mode bypassPermissions
codebuddy --permission-mode plan
```

```json
// 设置默认模式
{
  "permissions": {
    "defaultMode": "acceptEdits"
  }
}
```

## 1.5 headless模式

以编程方式运行 CodeBuddy Code，无需交互式 UI

## 1.6 从Claude Code迁移

### 迁移内容

| 目录/文件                    | 说明               |
| :--------------------------- | :----------------- |
| `agents/`                    | 自定义 agents 配置 |
| `commands/`                  | 斜杠命令定义       |
| `skills/`                    | 专业技能定义       |
| `CLAUDE.md` → `CODEBUDDY.md` | AI 指令和记忆文档  |

- 法一：符号链接，共享配置，修改一处两边生效。
- 法二：文件复制，独立配置，互不影响

```bash
# 验证迁移
codebuddy         # 启动
/skills           # 检查 Skills
/config           # 查看配置
```

## 1.7 成本优化

### 核心原则

- 新任务用 `/clear` 开启新会话
- 长对话用 `/compact` 压缩历史
- 用 `@filename` 引用文件，避免粘贴代码

### 会话管理命令

| 命令       | 功能            |
| :--------- | :-------------- |
| `/cost`    | 查看 Token 消耗 |
| `/clear`   | 开启新会话      |
| `/compact` | 压缩历史        |
| `/resume`  | 恢复旧对话      |

### 成本对比

| 方式                 | 输入 Token | 相对成本 |
| :------------------- | :--------- | :------- |
| 单会话连续 10 个任务 | ~50,000    | 高       |
| 每个任务新会话       | ~15,000    | 低       |
| 定期 `/compact`      | ~25,000    | 中       |

### 推荐做法

- ✓ 新任务开新会话
- ✓ 每 20-30 轮用 `/compact`
- ✓ 用 `@filename` 引用文件
- ✓ 精简提问

### 避免做法

- ✗ 同一会话处理多个无关任务
- ✗ 对话超过 30 轮不清理
- ✗ 重复粘贴已知代码

## 1.8 最佳实践

### 1.8.1 探索规划后编码

1. 探索：进入计划模式。CodeBuddy Code 只读取文件、回答问题，不做任何修改。
2. 规划：让 CodeBuddy Code 输出详细的实现计划
3. 编码实现：切换回普通模式，让 CodeBuddy Code 按计划编码，边做边验证。

### 配置我的环境

#### 首选语言

```bash
> /config
# 选择 Language，输入您的首选语言，如"简体中文"

# or ~/.codebuddy/settings.json
{
  "language": "简体中文"
}
```

#### CODEBUDDY.md

`/init` 命令会分析您的代码库，检测构建系统、测试框架和代码模式，为您生成一个可以继续完善的基础版本。

CODEBUDDY.md 没有固定格式，保持简短、易读就好。

CODEBUDDY.md 每次会话都会加载，所以只放那些普遍适用的内容。

CODEBUDDY.md 支持用 `@path/to/file` 语法导入其他文件

CODEBUDDY.md 可以放在多个位置

# 2 配置

CodeBuddy Code 使用分层配置系统，让您能够在不同级别进行个性化定制，从个人偏好到团队标准，再到项目特定需求。

## 2.1 配置文件

`settings.json` 文件是配置 CodeBuddy Code 的官方机制，支持分层设置：

- **用户设置** 定义在 `~/.codebuddy/settings.json`，应用于所有项目
- 项目设置存在项目目录中：
  - `.codebuddy/settings.json` 用于检入源代码控制并与团队共享的设置
  - `.codebuddy/settings.local.json` 用于不检入的设置，适合个人偏好和实验。CodeBuddy Code 会自动配置 git 忽略此文件

示例：

```json
{
  "language": "简体中文",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test:*)",
      "Read(~/.zshrc)"
    ],
    "ask": [
      "Bash(git push:*)"
    ],
    "deny": [
      "Bash(curl:*)",
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)"
    ]
  },
  "env": {
    "NODE_ENV": "development",
    "DEBUG": "codebuddy:*"
  },
  "model": "gpt-5",
  "cleanupPeriodDays": 30,
  "includeCoAuthoredBy": false,
  "statusLine": {
    "type": "command",
    "command": "~/.codebuddy/statusline.sh"
  }
}
```

## 2.2 配置项

- 权限配置
- 记忆：可以跨会话记住您的偏好
- bash sandbox
- 优先级
- 子代理
- 环境变量
- 状态行

## 2.3 配置管理命令

```bash
codebuddy config [command] [options]
```

## 2.4 记忆

CodeBuddy Code 提供四种分层结构的记忆位置，每种都有不同的用途：

| 记忆类型             | 位置                                            | 用途                           | 使用场景示例                     | 共享范围                     |
| :------------------- | :---------------------------------------------- | :----------------------------- | :------------------------------- | :--------------------------- |
| **用户记忆**         | `~/.codebuddy/CODEBUDDY.md`                     | 适用于所有项目的个人偏好       | 代码风格偏好、个人工具快捷方式   | 仅限本人（所有项目）         |
| **用户规则**         | `~/.codebuddy/rules/*.md`                       | 模块化的个人规则               | 个人编码习惯、常用工作流         | 仅限本人（所有项目）         |
| **项目记忆**         | `./CODEBUDDY.md` 或 `./.codebuddy/CODEBUDDY.md` | 项目的团队共享指令             | 项目架构、编码标准、常用工作流程 | 通过源代码管理与团队成员共享 |
| **项目规则**         | `./.codebuddy/rules/*.md`                       | 模块化的、按主题划分的项目指令 | 语言特定指南、测试规范、API 标准 | 通过源代码管理与团队成员共享 |
| **项目记忆（本地）** | `./CODEBUDDY.local.md`                          | 个人的项目特定偏好             | 您的沙箱 URL、首选测试数据       | 仅限本人（当前项目）         |

所有记忆文件在启动 CodeBuddy Code 时自动加载到上下文中。加载顺序如下：

1. **用户级**：加载 `~/.codebuddy/CODEBUDDY.md` 等主文件及 `~/.codebuddy/rules/` 下的所有规则
2. **项目级主文件**：从当前工作目录向上递归加载所有 `CODEBUDDY.md` 和 `CODEBUDDY.local.md`
3. **项目级规则**：仅加载当前工作目录的 `.codebuddy/rules/` 下的规则（不加载父目录的规则）
4. **子目录记忆**：当 CodeBuddy 操作子目录中的文件时，动态加载该子目录的 `CODEBUDDY.md`
5. **本地记忆**：加载 `./CODEBUDDY.local.md`

> **提示**：CODEBUDDY.local.md 文件会自动添加到 .gitignore，非常适合存储不应提交到版本控制的私有项目特定偏好。



/memory 管理记忆
