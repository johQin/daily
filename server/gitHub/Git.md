# GIT

# 2 GIT 基础

## 2.1 获取GIT仓库

### 2.1.1 初始化仓库init

跟踪管理项目文件

```bash
git init
```

### 2.1.2 克隆仓库clone

```bash
# 克隆远程库
git clone remote_address
# 克隆远程库的某个分支
git clone -b remote_branch_name remote_address
```

## 2.2 记录变更

- 工作目录下，每一个文件都处于两个状态
  - tracked，已跟踪，指上一次快照中包含的文件
    - 这些文件又可以分为：未修改，已修改，已暂存三个状态
  - untracked，未跟踪，除已跟踪文件之外的文件

### 2.2.1 本地库状态status

```bash
# 查看本地库详细的文件状态
git status
On branch feature-20210826-medIntercom
Your branch is up to date with 'origin/feature-20210826-medIntercom'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   src/page/medicalIntercom/corridorManagement/screenDisplay/api/factory.js

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   config/index.js
        modified:   static/color.less

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        src/page/medicalIntercom/corridorManagement/screenDisplay/ha.txt

# 简洁查看文件状态，s——short，simple
git status -s
## 开头有两列标记，左列标明了文件是否已暂存，右列标明文件是否被修改
 M config/index.js
M  src/page/medicalIntercom/corridorManagement/screenDisplay/api/factory.js
 M static/color.less
?? src/page/medicalIntercom/corridorManagement/screenDisplay/ha.txt

```

### 2.2.2 跟踪与暂存add

```bash
# 将未跟踪的文件或文件目录（递归跟踪）添加到已跟踪，untrack->tracked
git add filepath
git add filepath/file

# 将已修改文件添加到暂存区
git add filepath/file
```

### 2.2.3 差异分析diff

很多工具都可以为我们做可视化的差异分析，如果通过命令来看的话，那么可以使用diff

```bash
# 比较本地库中某个版本与工作区文件的差别，不加文件名比较所有文件
git diff version [file]
# 比较暂存区与本地库比较，staged与cached同义
git diff --staged [file]
git diff --cached [file]
```

### 2.2.4 提交变更commit

```bash
# 提交暂存区文件到本地库
git commit -m 'comment'
# 提交暂存区指定文件到本地库
git commit filepath/file -m 'comment' 
# 查看修改内容，并写comment，然后提交到本地库。v——visible
# 此时会打开vim，pageUp pageDown上下翻页，查看修改的地方，英文输入i，填写comment，“ esc :wq”退出vim完成提交,
git commit -v
# 将工作区内容直接提交到本地库，跳过暂存区
git commit -a -m 'comment'
```

### 2.2.5 文件操作

```bash
# 移除文件，删除文件并不再被git跟踪管理，git rm = delete file + git commit
git rm fliepath/[file]
# -f，可以强制将暂存区的对应文件也删除。

# 移除文件的git跟踪管理
git rm --cached filepath/[file]

# 移动（重命名）文件
git mv filepath_from/file_from filepath_to/file_to
```

### 2.2.6 忽略文件

忽略某些文件夹或文件，使其不被我们的git版本管理工具管理

通常可以配置`.gitignore`，具体的配置内容可以网络检索此文件的配置

## 2.3 日志log

```bash
# 以单行的形式输入提交日志
git log --pertty=oneline
# -p显示每次提交所引入的差异，-3显示最近的3次提交
git log -p -3
# 显示每次提交的简要统计
git log --stat
# 输出范围的选项，--since，--until
# 输出格式的选项format，等等
git log --pretty=format:"%H -%an"

# 按提交信息来过滤提交记录
git log --grep="【BUG修复】"
# 按时间段过滤提交记录
git log --after="2014-7-1" --before="2014-7-4"
# 查看文件的变更历史
git log foo.py
# 以提交者的名字查看提交历史
git log --author="khq"
#git还有很多关于输出范围和格式的选项，要用到时可以查
```

## 2.4 撤销操作

```bash
# 补交上一次提交的内容，修改文件，或修改提交信息comment，只产生上一次提交的记录，并且会提交在暂存区的内容
git commit --amend
# 撤销在暂存区中的内容
git reset HEAD [filepath/file]
# 撤销对文件的修改,恢复工作区文件为上一个版本
git checkout -- [filepath/file]
```

## 2.5 远程库remote

```bash
# 查看本地库对应的多个远程库地址
git remote -v
# 添加远程库地址
git remote add [shortname] [remote_address]
# 本地库具有远程库的跟踪分支的拉取和提交
git pull
git push
# 删除远程库和重命名
git remote rm [shortname]
git remote rename [cur_name] [will_name]
```

## 2.6 标记tag

为特定的历史版本打标记。

git的标记有两种：

- 轻量标签，可以作为某次提交的指针
- 注释标签，可以添加备注信息

```bash
# 查看标签
git tag
# 查看带有标签前缀的标签
git tag -l "标签前缀"

# 创建标签
# 创建轻量标签
git tag label
# 创建注释标签
git tag -a label -m "other_comment"

# 追加注释标签，为某个版本version_code追加tag
git tag -a label version_code

# 推送标签到远程库
git push origin [tagname] #有点像推送分支
git push origin --tags #一次性推送多个标签
# 执行完上述命令后，其他开发者在pull的时候，就会拉下tag信息。
```

## 2.7 git 别名

```bash
# 配置后，git co = git checkout
git config --global alias.co checkout
# 配置后，git last = git log -1 HEAD
git config --global alias.last 'log -1 HEAD' 
```

# 3 分支branch

在git add时，会为每个文件计算校验和（SHA-1散列值），当执行git commit 进行提交时，git会为每个子目录计算校验和，然后再把这些树对象保存到Git仓库中。

每次提交都会产生几个对象：blob对象（分别保存了项目的文件内容），1个记录着目录结构以及blob对象和文件名之间对应关系的树对象，1个包含提交全部元数据和指向根目录书对象的指针。

提交对象GIT的分支只不过是一个指向某次提交的可移动指针。

## 3.1 增加切换删除

创建新的分支时，git会创建可移动的新指针供你使用。

Git维护着一个名为HEAD的特殊指针，它是一个指向当前所在本地分支的指针。

分支的切换会更改工作目录的文件。在你切换分支之前要注意，如果你的工作目录或暂存区存在未提交的更改，并且这些更改与你要切换的分支冲突，git就不允许你切换分支（切换分支时，最好是保持一个干净的工作区域，后面会介绍绕开这个问题的办法：储藏stash和修订提交）

```bash
# 查看分支列表
git branch -a # 所有分支包括远程和本地，绿色字体是本地分支，红色字体是远程分支，加星号的是当前所在分支
# 查看本地分支
git branch
# 新建分支
git branch branch_name
# 从某次提交上新建分支
git branch branch_name version_SHA-1
# 切换分支管
git checkout branch_name
# 新建并切换分支
git checkout -b branch_name
# 删除分支，如果分支包含了尚未合并到主线的工作，-d不能删除该分支，但-D是可以强制删除的
git branch -d branch_name
```

## 3.2 合并merge

### 3.2.1 快进合并

当你试图去合并两个不同的提交，而顺着一个提交的历史可以直接到达另一个提交时，git就会简化合并操作，直接把分支指针向前移动，因为这种单线历史不存在分歧的工作，这就叫做`fast-forward`

### 3.2.2 三方合并

当两个分支出现了分叉，三方合并操作会使用两个待合并分支上最新提交的快照，以及这两个分支最近共同祖先的提交快照，基于这三方创建一个新的快照

#### 基本和合并冲突解决

当你在要合并的两个分支上都改了同一个文件的同一部分内容，git就没法干净地合并这两个分支。

```bash
$ git merge bug_fix1
Auto-merging index.html
CONFILCT(content)：Merge confilct in index.html
Automatic merge failed;fix conflicts and then commit the result.

$ git status
On branch master
You have unmerged paths
(fix conflicts and run "git commit")
Unmerged paths:
(Use "git add <file> ..." to mark resolution)

both modified:index.html
```

任何存在着未解决的合并冲突文件都会显示成未合并状态。git会给有冲突的文件添加标准的待解决冲突标记，以便你手动打开这些文件来解决冲突

删除冲突标记（<<<<  >>>>>)，并手动处理冲突部分的文件内容，执行`git add`将文件标记为冲突已解决状态，再执行`git commit `完成此次合并。

```bash
# 查看当前所有分支的简短列表
git branch
# 查看当前所有分支的最新提交列表
git branch -v
# 查看已并入和未并入当前分支的所有分支
git branch --merged
git branch --no-merged
```

## 3.3 远程分支

远程分支的表示形式：`remote_repository_shortname/remote_branch_name`    eg：`origin/master`

`master`被广泛使用只是因为它是执行`git init`时创建的初始分支的默认名称。

`origin`一样是`git clone`时远程仓库的默认名称，如需修改，可以采用`git clone -o replace_origin_name`

### 3.3.1 推送push

```bash
# 把当前分支的内容推送到远程库的某个远程分支
git push remote_repository_short_name remote_branch_name
# 把本地某个分支的内容推送到远程库的某个分支
git push remote_repository_short_name local_branch_name:remote_branch_name
```

```bash
# 拉取远程库的新内容
git fetch origin
# 合并新分支的内容到当前分支
git merge origin/new_branch_name
```

推送新的分支后，下一次与你协同的同事从服务器拉取数据时，他就会获取到一个指向服务器新分支的指针，这个指针就叫做`origin/new_branch_name`/。

注意：当获取服务器上的数据时，如果获取到本地还没有的新的远程跟踪分支，这是git并不会自动提供给你该分支的本地可编辑副本。换句话说，本地只是拥有了指向`origin/new_branch_name`的指针，不能直接作出修改。

```bash
# 拉取远程库新分支的内容，作为本地新建分支的内容
git checkout -b local_branch_name origin/new_branch_name
```

### 3.3.2 跟踪分支

基于远程分支创建的本地分支会自动成为跟踪分支（`tracking branch`），或者也叫做上游分支（`upstream branch`）

跟踪分支是与远程分支直接关联的本地分支。当你执行`git push`，git会知道推送到远程的哪个分支。当你执行`git pull`时，git也能知道从哪个服务器拉取数据，合并到本地分支

跟踪方式：

```bash
# 创建与远程分支同名的本地分支，会自动建立跟踪关系
git checkout local_branch_name_same_with_remote_branch_name
# 创建分支，并指定与远程分支的跟踪关系
git checkout -b local_branch_name origin/remote_branch_name
```

修改跟踪关系：

```bash
# 给本地已存在的分支设置跟踪关系、或要更改本地分支对应的远程分支的跟踪关系
# 使用-u 或 --set-upstream-to
git branch -u origin/remote_branch_name
git branch --set-upstream-to origin/remote_branch_name
```

查看跟踪关系：

```bash
git branch -vv
feature-20210726-medmrm 7c818f9 [origin/feature-20210726-medmrm: behind 4] 【功能新增】新增国际化错误码
* feature-20211026-medmrm 74aaa82 [origin/feature-20211026-medmrm] 【BUG修复】error_code维护，先结果后原因
```

注意：跟踪关系是从上次从远程服务器读取数据后开始计算的，这条命令并不会与服务器通信获取最新信息。

### 3.3.3 拉取fetch

`git fetch`命令会拉取本地没有的远程所有最新更改数据，但完全不会更改你的工作目录。

`git pull = git fetch + git merge`

### 3.3.4 本地操作远程分支

```bash
# 新建远程分支，提交本地local_branch_name分支作为远程remote_branch_name的分支
git push origin local_branch_name:remote_branch_name
# 删除远程分支
git push origin :remote_branch_name
git push origin --delete remote_branch_name
```

## 3.4 变基rebase

变基操作就是把某条开发分支线在另一条分支线上按顺序重现。而合并操作则是找出两个分支的，并他们合并到一起。

```bash
# 指定分支进行变基
git rebase base_branch_name topic_branch_name
# 将当前分支变基到指定分支
git rebase base_branch_name
```

变基后，通常还伴随着快进合并。

注意：仅对存在于本地还没有公开的提交进行变基操作。

# 4 git工具

## 4.1 引用版本

### 4.1.1 单个历史版本

1. SHA-1散列值

2. 短格式的SHA-1散列值：不少于4个且无歧义

   ```bash
   # 在保证不出现重复的情况下，命令会采用更简短的值
   git log --abbrev-commit
   ```

3. 分支引用，通过分支名引用分支名指向的版本

4. reflog：仓库操作日志，reflog信息智慧存在本地，通过`HEAD@{n}`，或者`branch_name@{n}`引用版本

   ```bash
   git reflog
   458cbba (HEAD -> dev, origin/dev-2021) HEAD@{0}: commit: 【BUG修复】保活请求参数未定义
   ebb8812 HEAD@{1}: pull: Fast-forward
   d6a1105 HEAD@{2}: commit: 【功能修改】保活处理
   40b1ab6 HEAD@{3}: pull: Fast-forward
   bb53dea HEAD@{4}: commit: 【BUG修复】导出loading恢复为初始态
   ebfb6fb HEAD@{5}: commit: 【BUG修复】导出增加loading态
   3292eaf HEAD@{6}: commit: 【bug修复】星期数组误用为函数，导致页面异常
   f41ae40 HEAD@{7}: pull: Merge made by the 'recursive' strategy.
   ```

5. 祖先引用

   - `^`
     - `^^`：首个父提交的首个父提交，祖父提交，`^^^`：祖父提交的父提交
     - `^2`：当前提交的第二个父提交，仅当有合并的时候，才会存在一个版本有两个父提交
   - `~`
     - `~~`：首个父提交的首个父提交，祖父提交，`~~~`：祖父提交的父提交
     - `~2`：首个父提交的首个父提交，祖父提交，`^3`：祖父提交的父提交
   - **branch_name、HEAD、散列值都可以用祖先引用**

```bash
# git show 用于显示各种类型的对象，树，提交对象，分支对象。

# branch_name、HEAD都可以替换
git show branch_name@{1}
git show HEAD@{2}
# 在祖先引用中，branch_name、HEAD、散列值都可以替换
git show branch_name^
git show branch_name~
git show branch_name^^^
git show branch_name~~~
git show branch_name^~
```

### 4.1.2 引用版本范围

在分支管理中，指定特定的范围的版本。

```bash
# 两个点，比较两个引用，只列出后者的差异提交
git log master..experiment
# 三个点，比较两个引用，列出二者的差异提交
git log [--left-right] master...experiment

eg:
HEAD~3..HEAD
```

## 4.2 交互式处理暂存区

在许多命令中，都存在`-i`用于交互式

```bash
git add -i
		staged     unstaged path
  1:    unchanged        +1/-1 config/dev.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/js/securityLevel.js
*** Commands ***
  1: status       2: update       3: revert       4: add untracked
  5: patch        6: diff         7: quit         8: help
What now>2 [enter]
# 选择命令，通过数字或命令首字母
# status，和git add -i命令的功能一样
# update，将文件添加到暂存区，暂存区也称索引库
# revert，将已经添加到暂存区的文件从暂存区中删除
# add untracked，可以把没被git管理的文件添加到索引库中
# patch，暂存补丁，只暂存文件的一部分
# diff，可以比较索引库中文件和原版本的差异
# quit，退出git add -i
# help，可以查看git add的帮助文档，与git add -h功能一致
           staged     unstaged path
  1:    unchanged        +1/-1 config/dev.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
Update>> 1,2 [enter]
# 选择文件，需要通过文件前面的编号进行选择，
# 1,3选择编号不连续的文件（2不选），1-3选择编号连续的文件
           staged     unstaged path
* 1:    unchanged        +1/-1 config/dev.js # * 表示该文件已在暂存区（索引库）
* 2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
Update>>[enter]
updated 2 paths

*** Commands ***
  1: status       2: update       3: revert       4: add untracked
  5: patch        6: diff         7: quit         8: help
What now> 3
Revert>> 1-2
           staged     unstaged path
* 1:        +1/-1      nothing config/dev.js
* 2:      +23/-28      nothing config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
Revert>>
reverted 2 paths

*** Commands ***
  1: status       2: update       3: revert       4: add untracked
  5: patch        6: diff         7: quit         8: help
What now> 1
           staged     unstaged path
  1:    unchanged        +1/-1 config/dev.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js

```

### 暂存补丁

git也可以只暂存文件的某些部分。

```bash
git add -i
		staged     unstaged path
  1:    unchanged        +1/-1 config/dev.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
*** Commands ***
  1: status       2: update       3: revert       4: add untracked
  5: patch        6: diff         7: quit         8: help
What now>5
           staged     unstaged path
  1:    unchanged        +1/-1 config/dev.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
Patch update>> 1
           staged     unstaged path
* 1:    unchanged        +1/-1 config/dev.env.js
  2:    unchanged      +23/-28 config/index.js
  3:    unchanged    +207/-174 src/common/utils.js
Patch update>>[enter]
diff --git a/config/dev.env.js b/config/dev.js
index ecbf751..9d08246 100644
--- a/config/dev.env.js
+++ b/config/dev.env.js
@@ -11,5 +11,5 @@ const merge = require("webpack-merge");
 const prodEnv = require("./prod.env");
 module.exports = merge(prodEnv, {
   NODE_ENV: '"development"',
-  WS_API: '"10.80.3.54"'
+  WS_API: '"10.80.3.123"'
 });
Stage this hunk [y,n,q,a,d,e,?]? n
# y - 暂存此区块
# n - 不暂存此区块
# q - 退出；不暂存包括此块在内的剩余的区块
# a - 暂存此块与此文件后面所有的区块
# d - 不暂存此块与此文件后面所有的 区块
# g - 选择并跳转至一个区块
# / - 搜索与给定正则表达示匹配的区块
# j - 暂不决定，转至下一个未决定的区块
# J - 暂不决定，转至一个区块
# k - 暂不决定，转至上一个未决定的区块
# K - 暂不决定，转至上一个区块
# s - 将当前的区块分割成多个较小的区块
# e - 手动编辑当前的区块
# ? - 输出帮助
```

## 4.3 储藏stash

通常当你在处理项目的某一部分是，方方面面的事情还未处理完成，你想转入其他分支忙别的事情，此时你也不希望把先前只做了一半的工作提交，因为随后还想回过头接着做。

储藏（stash）能够获取工作目录的中间状态，也就是修改过的被跟踪的文件以及暂存的变更，并将该中间状态保存在一个包含未完成变更的栈中，随后可以再恢复这些状态。

### 4.3.1 基本

```bash
# 储藏变更。此时打算切换分支，又不想把变更提交，git stash save 将新的储藏内容推入栈中
git stash save 'comment'
# 查看储藏栈列表
git stash list
stash@{0}: On feature-20210826: 20211115 1440
stash@{1}: On feature-20210826: 20211113 1109
stash@{2}: On feature-20210826: 20211111 1453
stash@{3}: On feature-20210826: 2021110 1448

# 应用栈中的储藏，但不出栈（删除）
git stash apply stash@{n} # n是列表中的数字
# 可以跨分支应用，但只要无法干净利落的应用任何操作，git都会给出合并冲突信息
# 应用栈中的储藏，但会被删除
git stash pop stash@{n}
# 删除储藏内容
git stash drop stash@{n}
```

### 4.3.2 从储藏中建立分支

`git stash branch branch_name stash@{n}`，该命令会为你创建一个新的分支，该分支是建立在你储藏成果`stash@{n}`时所在的提交之上，如果重新应用成功，那么就会丢弃掉储藏。

该命令的场景是：在储藏后的分支上继续做了大量的修改，此时应用储藏就会出现许多不得不解决的冲突。如果想要一种更简便的方法来重新检验储藏的变更，可以执行 git stash branch。

### 4.3.3 清理clean

`git clean`默认可以删除所有**未跟踪**且在`.gitignore`文件中**未被忽略**的文件

```bash
git clean -f -d -x -n
# f 强制删除
# n 预演需要删除那些文件，这个操作只会告诉你会删除什么文件，而不会真正删除这些文件。f 和 n不能同时使用
# x 会删除那些在.gitignore规定被忽略的文件
# d directory，目录
```

## 4.4 搜索

经常会查找某个函数的调用位置或定义位置，或者某个方法的变更历史。

```bash
# 在什么地方出现

git grep -n explore_content
# n，输出匹配位置的行号
git grep --count explore_content
# count，统计匹配到哪些文件，每个匹配的文件中有多少处匹配

# 在什么时候出现

# -S让Git只显示出添加过或删除过该字符串的提交
git log -S PROCESS_WSAPI --oneline
# -L让Git展示代码库中某个函数或代码行的历史
git log -L :function_name:file_name

```

## 4.5 重写历史

你可能出于某些原因想要修订提交历史，这些涉及到改变提交顺序，修改提交包含的消息或文件，压缩，拆分，完全删除提交。

这一切都可以再你尚未同他人共享工作成果之前进行。

前提是这些历史并未推送到中央服务器，也就是说只能修改未push的历史。

### 4.5.1 修改最近一次提交

```bash
touch 4.txt
git add 4.txt
git commit --amend
# 此条命令会获取你当前的暂存区并将它作为新提交的快照，所以4.txt会提交到当次提交当中
# 并且会改变提交的SHA-1值，所以不要在推送了最近一次提交之后还去修正它
```

### 4.5.2 交互式变基工具

借助交互式的变基工具，你可以在每个想要修改的提交停下来，然后改变提交消息、添加文件或是做任何想做的事。`git rebase -i`选项可以带你进入变基的交互模式

```bash
# 如果你想改变最近三次提交或其中任意一次的提交消息
# 需要倒数第三次提交的父引用作为参数传给git rebase -i
git rebase -i HEAD~3

# 进入文本编辑器
# 这是一个待运行的脚本，该脚本会从你在命令行指定的提交开始（HEAD~3)，自上而下地**重演**每个提交所引入的变更。它将最早的（而非最近的）提交列在顶端，因为这是第一个需要重演的
pick 7103862 3th commit
pick 2fc8ece 4th commit
pick 1c1960c 5th commit

# Rebase 73ed7c4..1c1960c onto 73ed7c4 (3 commands)
#
# Commands:
# p, pick <commit> = use commit
# 重排提交
# 代表commit命令，1.如果删除pick行，则代表不提交。2.调整pick行的顺序会改变提交的顺序，

# r, reword <commit> = use commit, but edit the commit message
# 修改提交的消息comment，而不会改变提交的内容

# e, edit <commit> = use commit, but stop for amending
# 修改提交的内容，可以做额外的添加、修改内容操作

# s, squash <commit> = use commit, but meld into previous commit
# f, fixup <commit> = like "squash", but discard this commit's log message
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with 'git rebase --continue')
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
# .       create a merge commit using the original merge commit's
# .       message (or the oneline, if no original merge commit was
# .       specified). Use -c <commit> to reword the commit message.
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out

```

### 4.5.3 修改多个提交历史edit

```bash
/d/develop/laboratory (master)
$ git rebase -i HEAD~3
# 在文本编辑器中，只对每行的首部子命令做了修改
# pick 7103862 3th commit
# edit fd1f865 4th commit
# edit 8bd653c 5th commit

# 会在edit的地方进行重演
Stopped at fd1f865...  4th edit commit
You can amend the commit now, with
# 做了相应的修改后，通过git commit --amend进行当前历史版本的提交
  git commit --amend

Once you are satisfied with your changes, run
# 当你在该版本已做完所有修改，可以执行git rebase --continue想到下一个edit的历史版本做重演
  git rebase --continue

# 你可以看见分支名这里发生了改变，（master|REBASE-i 2/3)
# 这表示你正在第几个版本进行历史的重演，fd1f865


/d/develop/laboratory (master|REBASE-i 2/3)
$ git commit --amend -m '4th edit2 commit'
# 对当前的历史版本进行修改，你可以修改文件然后add，然后commit --amend
# 执行amend之后，该历史版本的SHA-1发生了变化，4baac78
[detached HEAD 4baac78] 4th edit2 commit
 Date: Thu Dec 9 10:35:28 2021 +0800
 2 files changed, 1 insertion(+)
 create mode 100644 4.txt
 create mode 100644 c.txt

 /d/develop/laboratory (master|REBASE-i 2/3)
$ git rebase --continue
Stopped at 8bd653c...  5th commit
You can amend the commit now, with

  git commit --amend

Once you are satisfied with your changes, run

  git rebase --continue

/d/develop/laboratory (master|REBASE-i 3/3)
$ git commit --amend '5th edit commit'
error: pathspec '5th edit commit' did not match any file(s) known to git

/d/develop/laboratory (master|REBASE-i 3/3)
$ git commit --amend -m '5th edit commit'
[detached HEAD 00f04ae] 5th edit commit
 Date: Thu Dec 9 10:45:59 2021 +0800
 1 file changed, 2 insertions(+)

/d/develop/laboratory (master|REBASE-i 3/3)
$ git rebase --continue
Successfully rebased and updated refs/heads/master.

/d/develop/laboratory (master)
```

### 4.5.4 重排提交

删除pick行，就不提交该历史版本。

调整pick行的顺序，就可以修改提交版本的顺序。

### 4.5.5 压缩提交squash

将一系列提交压缩成单个提交。

```bash
/d/develop/laboratory (master)
$ git log --oneline
b93d434 (HEAD -> master) 5th edit commit
192b9cb 3th and 4th compress commit
73ed7c4 2th commit
842c1a1 1th commit

# 文本编辑器中
# pick 73ed7c4 2th commit
# squash 192b9cb 3th and 4th compress commit
# squash b93d434 5th edit commit

# 然后在压缩编辑文本器中编辑提交的消息
# This is a combination of 3 commits.
# This is the 1st commit message:

2th 3th 4th 5th comporess commit

# This is the commit message #2:

# This is the commit message #3:


# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Date:      Thu Dec 9 10:23:03 2021 +0800
#
# interactive rebase in progress; onto 842c1a1
# Last commands done (3 commands done):
#    squash 192b9cb 3th and 4th compress commit
#    squash b93d434 5th edit commit
# No commands remaining.
# You are currently rebasing branch 'master' on '842c1a1'.
#
# Changes to be committed:
#       new file:   .gitignore
#       new file:   4.txt
#       new file:   b.txt
#       new file:   c.txt
#

# 压缩成功
[detached HEAD ab5df8a] 2th 3th 4th 5th comporess commit
 Date: Thu Dec 9 10:23:03 2021 +0800
 4 files changed, 27 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 4.txt
 create mode 100644 b.txt
 create mode 100644 c.txt
Successfully rebased and updated refs/heads/master.

/d/develop/laboratory (master)
$ git log --oneline
ab5df8a (HEAD -> master) 2th 3th 4th 5th comporess commit
842c1a1 1th commit
```

### 4.5.6 拆分提交

将需要拆分的历史提交版本修改为`edit`。

然后进行多次`git commit`而不是`--amend`，这样一个提交就拆分成了多个提交。

最后再进行`git rebase --continue`

### 4.5.7 重写大量历史filter-branch

前提：你的项目未公开，也没有人在你打算提交的基础上开展过工作。

filter-branch能够大面积修改你的历史记录。

#### 从所有提交中删除某个文件

filter-branch是一个可以用来清洗整个历史记录的工具。

```bash
git filter-branch --tree-filter 'rm -f passsword.txt' HEAD
# --tree-filter选项会在每次检出项目后执行指定命令，然后重新提交结果
```

## 4.6 重置reset

### 4.6.1 三棵树

也就是三个区域。

1. HEAD（版本库）

   - 它是指向当前分支最后一次提交的指针

2. 索引（暂存区）

   - git会将上次检出到工作目录的所有文件的列表填入索引

   - ```bash
     git commit -m ''
     # 将暂存区的文件提交到版本库
     ```

3. 工作目录（工作区），

   - 可以把工作目录当成沙盒，在将内容提交到暂存区并写入版本库前，可以随意修改

   - ```bash
     git add filepath/file
     #将工作区的文件进行追踪，并将修改提交到暂存区
     ```

### 4.6.2 重置的作用

```bash
# 仅移动HEAD指针到相应的版本引用
git reset --soft version_SHA-1
# 会清空当前的暂存区的内容，移动HEAD指针和暂存区到相应版本，
git reset [--mixed] version_SHA-1
# 会清空当前的暂存区和工作区的内容，移动HEAD指针、暂存区和工作区到相应版本
git reset --hard version_SHA-1
```

reset命令会以特定的次序重写这三棵树，并在你指定操作时停止

1. 移动HEAD分支的指向（指定了`--soft`才会这样做）
2. 使索引看起来像HEAD（不加任何指令，默认会执行这一条，或者加`--mixed`）
3. 使工作目录看起来像索引（指定`--hard`才会这样做）

### 4.6.3 重置+文件路径

如果在git reset后指定了文件的路径或路径加文件名，reset就会跳过上面步骤中的第1步，将剩余操作限定在特定的一个或一组文件。

也就是说`git reset [--mixed or --hard] [version_SHA-1] filepath/file`，HEAD指针并不会移动（跳过的第1步），而会按照指令`--mixed or --hard`重置暂存区or工作区中特定的一个或一组文件到某一版本。

### 4.6.4 删除中间的提交历史

只要看了上面的reset内容，你会立马明白下面命令的意义。

这里和重写历史中的`squash`还是有区别，squash的操作是将多个commit操作，按顺序揉在一个commit里面。

而这里的reset通过移动指针，后再提交，是将中间的版本丢弃掉，而不是squash的压缩。

```bash
# 移动HEAD指针到到特定引用版本，也可以用SHA-1
git reset --soft HEAD~n
# 将暂存区中的修改提交到HEAD~n之后的一个版本，这使得之前HEAD~n之后的版本被丢弃。
git commit -m ''
```

### 4.6.5 检出checkout与reset的区别

与reset一样，checkout也能操纵这三棵树，不过略有差异，这取决于你是否为该命令传入文件路径。

#### 不使用路径

`git checkout [branch]`和`git reset --hard [branch]`极为相似，但有两点不同。

1. 切换分支时，checkout会进行琐碎的合并，一旦有冲突，就不会切换。而reset --hard并不会检查，而只是简单的进行全面替换。一般用checkout切换分支，而通过reset移动当前分支的版本
2. 更新HEAD的方式，reset移动的是HEAD指向的分支（HEAD与分支的指针同时移动），而checkout只移动HEAD

#### 使用路径

加上文件路径时，与reset一样，不会移动HEAD。

## 4.7 调试

### 4.7.1 文件标注

显示出文件的每一行最后都是由哪一次提交修改的。

如果你知道问题出自哪里，那么标注文件就能发挥作用

```bash
# 查看文件每一行最后都是由哪一次提交修改的
git blame -- filepath/file
# 首部打^表示文件从跟踪到最新版本都没有被修改过
^842c1a1 (qkh 2021-12-09 10:21:41 +0800 1) 你好呀世界 
ab5df8a7 (qkh 2021-12-09 10:23:03 +0800 1) 生命就像一条大河，时而宁静，时而疯狂。
ab5df8a7 (qkh 2021-12-09 10:23:03 +0800 2) 现实就像，一把枷锁，把我捆住，无法挣脱。

# 查看文件的在指定行范围内，每一行最后都是由哪一次提交修改的
git blame -L 12,22 filepath/file
```

### 4.7.2 二分查找

当你不知道问题在哪里，并且自上一次代码处于正常工作状态已经提交了上百次，可能就得求助于`git bisect`。bisect命令会对你的提交历史记录进行二分查找，**帮助尽快确定问题是由哪一次提交造成的**

```bash
# 启动排查流程
git bisect start
# 告知系统哪一次提交有问题，省去hash码，代表当前提交版本有问题
git bisect bad [version_SHA-1]
# 告知系统最后一次正常状态是什么时候，省去hash码，代表当前提交版本有问题
git bisect good [version_SHA-1]
# 恢复HEAD指针，或者离开排查流程
git bisect reset
```



```bash
/d/develop/laboratory (master)
$ git log --oneline
03f88db (HEAD -> master) c.txt 验证blame第三次修改
6557cd9 b.txt commit
187ea53 单方面提交c.txt，验证是否把暂存区所有文件都提交
ab5df8a 2th 3th 4th 5th comporess commit
842c1a1 1th commit


# 启动排查流程
/d/develop/laboratory (master)
$ git bisect start
# 告知系统当前版本有问题
 /d/develop/laboratory (master|BISECTING)
$ git bisect bad
# 告知系统最后一次正常的时候是哪一次提交
/d/develop/laboratory (master|BISECTING)
$ git bisect good 842c1a1
Bisecting: 1 revision left to test after this (roughly 1 step)
[187ea53ba1c0bb03eb7a218f6122b5ce50e4a642] 单方面提交c.txt，验证是否把暂存区所有文件都提交

/d/develop/laboratory (master|BISECTING)
$ git log --oneline
# 可以从这里看到排查的开始点，也就是二分法的开始排查的地方，这里是坏的
03f88db (HEAD -> master, refs/bisect/bad) c.txt 验证blame第三次修改
6557cd9 b.txt commit
187ea53 单方面提交c.txt，验证是否把暂存区所有文件都提交
ab5df8a 2th 3th 4th 5th comporess commit
# 这里是好的
842c1a1 (refs/bisect/good-842c1a1b2b1ab53b91a051753788cd6d8b4b9dd6) 1th commit

# 告知系统当前版本是坏的，
/d/develop/laboratory ((187ea53...)|BISECTING)
$ git bisect bad
Bisecting: 0 revisions left to test after this (roughly 0 steps)
[ab5df8a741a96968ef3c2eb9d53de6763ba5010a] 2th 3th 4th 5th comporess commit
# 告知系统当前版本是好的
/d/develop/laboratory ((ab5df8a...)|BISECTING)
$ git bisect good
# 二分法结束，排查出结果
187ea53ba1c0bb03eb7a218f6122b5ce50e4a642 is the first bad commit
commit 187ea53ba1c0bb03eb7a218f6122b5ce50e4a642
Author: qkh
Date:   Fri Dec 10 10:25:02 2021 +0800

    单方面提交c.txt，验证是否把暂存区所有文件都提交

 c.txt | 2 ++
 1 file changed, 2 insertions(+)

/d/develop/laboratory ((ab5df8a...)|BISECTING)
$ git log --oneline
ab5df8a (HEAD, refs/bisect/good-ab5df8a741a96968ef3c2eb9d53de6763ba5010a) 2th 3th 4th 5th comporess commit
842c1a1 (refs/bisect/good-842c1a1b2b1ab53b91a051753788cd6d8b4b9dd6) 1th commit

/d/develop/laboratory ((ab5df8a...)|BISECTING)
$ git bisect reset
Previous HEAD position was ab5df8a 2th 3th 4th 5th comporess commit
Switched to branch 'master'

/d/develop/laboratory (master)
$
```

## 4.8 替换

假设有一份体积庞大的历史记录，希望将仓库划分成两部分：较短的历史记录留给新的开发者使用，较久远的历史记录留给对数据挖掘感兴趣的用户使用。

我们可以通过用旧仓库中的最新提交替换新仓库的最旧提交来实现历史的嫁接。

### 历史划分

1. 通过commit-tree命令创建一个全新的、无父节点的提交对象
2. 再通过`rebase --onto`命令将余下的历史记录变基到基础提交（也就是刚刚创建的对象）之上

```bash
/d/develop/laboratory (master)
$ git log --oneline
d0702d4 (HEAD -> master) 8th commit
a6cea54 7th commit
495d19a 6th commit
03f88db c.txt 验证blame第三次修改
6557cd9 b.txt commit
187ea53 单方面提交c.txt，验证是否把暂存区所有文件都提交
ab5df8a (history) 2th 3th 4th 5th comporess commit
842c1a1 1th commit

/d/develop/laboratory (master)
echo 'get history from new commit Object' | git commit-tree ab5df8a^{tree}
2212a11e065e3d18f1c2cd330f27ec065038a682

/d/develop/laboratory (master)
$ git rebase --onto 2212a11e ab5df8a
First, rewinding head to replay your work on top of it...
Applying: 单方面提交c.txt，验证是否把暂存区所有文件都提交
Applying: b.txt commit
Applying: c.txt 验证blame第三次修改
Applying: 6th commit
Applying: 7th commit
Applying: 8th commit
```

### 恢复历史



```bash
git replace new_commit old_commit
```

## 4.9 子模块

子模块（子仓库）允许你将一个Git仓库作为另一个Git仓库的子目录，这样你就可以将其他仓库克隆到你的项目中，同时保持提交的独立性。

### 4.9.1 创建子模块

```bash
# 创建子模块
git submodule add <url> <path>
# 其中，url为子模块的路径，path为该子模块存储的目录路径。
# 执行成功后，git status会看到项目中修改了.gitmodules，并增加了一个新文件（为刚刚添加的路径）
```

尽管新文件是工作目录下的一个子目录，但git将其视为一个子模块，当你不在该新文件夹中时，Git并不会跟踪其中的内容。

### 4.9.2 克隆含有子模块的项目

如果简单实用`git clone url`，默认会得到子模块的目录，但是目录中并不会有子仓库的文件。所以要使用到--recursive参数

```bash
# --recursive选项，会自动初始化和更新父仓库father_url下的每一个子模块。
git clone --recursive father_url
```

如果在子模块的文件夹中，打开命令行，那么就可以像平常操作仓库一样对子仓库进行操作。

# 工作场景

## 一般工作场景

提交代码到远程库

```bash
# 1. 提交自己的文件到本地库之前，需要将远程库的内容更新到本地，减少后面不必要的三方merge操作
git pull
# 2. 将工作区文件修改提交到暂存区
git add filepath/file
# 3. 提交暂存区文件到本地库
git commit -m ''

# 如果第1步和第3步间隔时间较长那么建议再执行一次git pull

# 4. 将本地库最新版本的内容提交到远程库
git push

# 冲突场景解决
# 一旦pull时，发生自动merge失败，也就是说发生冲突
# 1. 暂存工作区文件的修改，然后再pull
git stash save ''
# 2. 将修改弹出，这时必然会造成冲突conflict，去相关的冲突文件内，将《《《《 》》》标记的区域的冲突内容修改掉，冲突的内容就会消失
git stash pop stash@{0}
# 3. 文件中解决冲突后，需要手动add修改后的文件，这样需要merge的文件就消失了。

# 重置文件
git reset advdcdcecdccc filepath/file
git reset HEAD -- filepath/file # 重置暂存区的某个文件

git checkout -- filepath/file # 从暂存区（暂存区如果没有，则从本地库）恢复最新版本的该文件到工作区
git checkout version_SHA-1 filename
```

```bash
# 查看配置项
git config list
```

# 其他知识点

## 1 git设置提交模板

1. 在某一个固定的文件夹下，新建一个Git提交的模板文本文件

   - 例如在`.ssh`文件夹下，新建`.gitmessage.txt`文件，

   - `.gitmessage.txt`的内容就是每次提交想要的模板

   - ```
     【功能修改】具体修改
     【开发周期】H
     【提交人】rrr(工号)
     ```

2. gitconfig生效修改

   - 执行命令

     ```bash
     git config –global commit.template /c/Users/具体user/.ssh/.gitmessage.txt
     ```

   - 或者打开.gitconfig文件，添加配置内容

     ```
     [commit]
     	template = C:/Users/具体user/.ssh/.gitmessage.txt
     ```

     

3. 
