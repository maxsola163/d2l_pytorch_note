# Git 笔记

## 配置作者信息

```bash
# 全局
git config --global user.email "<email_url>"
git config --global user.name "<name>"

# 当前项目
git config user.email "<email_url>"
git config user.name "<name>"
```

> 配置文件在 `.gitconfig`

## 仓库管理

**创建仓库**

```bash
# 创建新仓库
git init
# 克隆已有仓库
git clone <url>
```

**流水操作**

```bash
# 添加文件（暂存区）
git add <file_name>
# 提交修改
git commit -m '<commit_comment>'
# 仓库状态
git status
```

**忽略文件**

```bash
# .gitignore
*.txt			
!a.txt			# 除了 a.txt
/folder_name
/folder_name/**	# 忽略子目录
```

**删除文件**

```bash
# 同时删除版本库和本地
git rm <file_name>
# 仅移除版本库
git rm --cached <file_name>
```

**修改名字**

```bash
git mv <old_file_name> <new_file_name>
```

**日志**

```bash
# 显示日志
git log
# 简单日志
git log --oneline
# 显示文件变动
git log -p
# 仅显示最近一次
git log -p -1
# 显示修改的文件
git log --name-only
# 显示文件，包含操作方式
git log --name-status
# 显示分支
git log --graph
```

**修改提交描述**

```bash
# 修改最后一次提交的描述
git commit --amend
```

**撤销修改**

```bash
# 撤销缓存操作
git restore --staged <file_name>
# 恢复到上一次修改
git restore <file_name>
```

**使用别名**

```bash
git config --global alias.a add
```

> 可以直接修改`.gitconfig`

```bash
# .gitconfig
[alias]
	a = add
	c = commit
	s = status
	l = log --oneline
```

**回退版本**

```bash
# 回退到之前的版本，且删除之后的版本
git reset --hard <hash_code>
# 反做，将之前的提交重新提交一次
git revert -n <hash_code>
```

## 分支

默认分支为主分支 `master`[快速合并 `Fast-forward`]

```bash
# 查看分支
git branch
# 创建分支
git branch <branch_name>
# 查看本地和远程分支关系
git branch -a
# 切换分支
git checkout <branch_name>
# 创建并且切换
git checkout -b <branch_name>
```

**合并与删除分支**

```bash
# 合并分支
git merge <branch_name>
# 删除分支
git branch -d <branch_name>
```

**解决冲突**

直接修改冲突文件后提交，然后再次合并

```bash
# 查看已经合并的分支
git branch --merged
# 查看没有合并的分支
git branch --no-merged
# 删除没有合并的分支
git branch -D <no_merged_branch_name>
```

## stash临时存储区

当缓冲区有文件时，无法切换分支。**可以对工作区的状态进行存储**

```bash
# 存储至存储区
git stash
# 查看存储区
git stash list
# 恢复存储区（不删）
git stash apply
# 恢复并删除
git stash pop
# 删除存储区
git stash drop stash@{...}
```

> 只有在 `add` 或 `commit` 之后才能放置到存储区

## tag 标签

```bash
# 打标签
git tag <tag_name>
# 查看标签
git tag
```

## 发布代码

```bash
git archive master --prefix='<fold_name>' --forma=zip > release.zip
```

## rebase

`rebase = replace rebase`：将子分支的提交点点移至主分支的最新点

```bash
git rebase master
```

## 远程库

**克隆远程库**

```bash
git clone <url>
```

**推送到远程库**

```bash
git push
git push -f 				# 强制推送
git push -u origin master	# 推送到远程服务器的 master 分支
```

**添加远程库**

```bash
# 关联远程库
git remote add origin <url>
```

**远程分支管理**

```bash
# 推送并且创建远程分支
git push --set-upstream origin <branch_name>
# 查看本地和远程分支关系
git branch -a
# 创建远程分支与本地分支的联系
git pull origin <branch_name>:<branch_name>
# 删除远程分支
git push origin --delete <branch_name>
```
