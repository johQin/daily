# [Jenkins](https://www.jenkins.io/zh/)

Jenkins是一款基于java开发的开源 CI&CD 软件，用于自动化各种任务，包括构建、测试和部署软件。

在典型的软件开发和测试流程中，Jenkins 通常处于开发阶段与测试阶段之间的桥梁位置，同时也贯穿于整个开发周期，具体如下：

1. 开发阶段：开发人员完成代码编写后，将代码提交到版本控制系统（如 Git）。Jenkins 通过监听版本控制系统中的代码提交事件，触发后续的自动化流程。
1. 构建阶段：Jenkins 在代码提交后，自动拉取代码并执行构建任务（如编译代码、运行单元测试等）。这一阶段是开发和测试的过渡环节，确保代码能够正常编译并且通过初步的单元测试。
1. 测试阶段：构建完成后，Jenkins 可以自动部署构建产物到测试环境，并触发自动化测试（如接口测试、功能测试等）。测试人员基于 Jenkins 提供的反馈，进行进一步的手动测试或验证。
1. 部署阶段：在测试通过后，Jenkins 可以将代码部署到生产环境，实现持续部署。它在整个开发和测试流程中起到了串联各个环节的作用。

主要内容：

1. 基础运行环境快速部署
2. Jenkins传统/Blue Ocean UI使用
3. 一键Maven拉取git代码完成构建jar包，并提交测试服务器自动运行
4. IDE提交代码后自动构建并发布任务
5. 定时构建发布任务
6. 邮件通知任务执行结果
7. Jenkins构建项目自动化运行在Docker容器中
8. Jenkins Pipeline脚本与Jenkinsfile使用
9. Jenkins 多分支项目

[Jenkins官方文档](https://www.jenkins.io/zh/doc/)

# 1 安装部署

## 1.1 安装gitlab

```bash
# 安装ssh perl
sudo apt install -y curl openssh-server perl
# 
```



在gitlab的整包中，包含但不限于：

- postgresql：为应用程序元数据和用户信息提供存储。
- puma：一个快速，多线程，高并发的http 1.1服务器，适用于Ruby程序，运行核心的Rails应用程序，提供极狐Gitlab的用户界面功能
- nginx - 作为http请求入口，将http请求路由到合适的gitlab子系统。
- redis - 主要用于以下目的：缓存、Sidekiq的作业处理队列、管理共享应用程序状态、存储 CI 跟踪块、作为 ActionCable 的 Pub/Sub 队列后端、速率限制状态存储、Session。
- sidekiq - Ruby应用程序。sidekiq是一个后台作业处理器，从Redis队列中提取作业并进行处理。

```bash
# 启动所有gitlab组件
gitlab-ctl start

# 停止所有gitlab组件
gitlab-ctl stop
# 重启所有gitlab组件
gitlab-ctl restart
# 查看服务状态
gitlab-ctl status
gitlab-ctl reconfigure

# 在首次执行gitlab-ctl start 后面有
# Notes:Default admin account has been configured with following details:Username: rootPassword: You didn't opt-in to print initial root password to STDOUT.Password stored to /etc/gitlab/initial_root_password. This file will be cleaned up in first reconfigure run after 24 hours.
#NOTE: Because these credentials might be present in your log files in plain text, it is highly recommended to reset the password following https://docs.gitlab.com/ee/security/reset_user_password.html#reset-your-root-password.

# 在这个文件中获取初始密码后，/etc/gitlab/initial_root_password
# 然后在浏览器中登录，然后再去改密码
```



## 1.2 Jenkins 安装

[下载](https://www.jenkins.io/zh/download/)，[机器配置](https://www.jenkins.io/zh/doc/book/installing/#prerequisites)

Jenkins 以 WAR 文件、原生包/安装程序和 Docker 镜像分发。

[如果通过war包安装，需要先安装jdk](https://www.jenkins.io/zh/doc/book/installing/#war%E6%96%87%E4%BB%B6)

```bash
# JENKINS_HOME 环境变量用于指定 Jenkins 的主目录位置，所有的 Jenkins 配置文件、工作目录以及插件等都存储在这个目录中。
# 默认位置（通常是 /var/lib/jenkins 或者其他根据安装方式不同的默认路径）。	
export JENKINS_HOME=/home/jenkins/jenkins_home
# 运行命令，如果在运行后，你中断这个程序，那么jenkins服务就会被关闭
java -jar jenkins.war

#
# Jenkins initial setup is required. An admin user has been created and a password generated.Please use the following password to proceed to installation:
# 4f8da39c58b64fbebbb7838a30a6fe026
# This may also be found at: /root/.jenkins/secrets/initialAdminPassword

# 浏览http://localhost:8080并等到*Unlock Jenkins*页面出现。

# 继续使用Post-installation setup wizard后面步骤设置向导。
```

