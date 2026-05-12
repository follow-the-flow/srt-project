# Git Push 教程

## ⚠️ 前置准备（只需做一次）

**第零步：新建一个空文件夹，专门用于 git push**

> 建议不要直接在已有项目文件夹里gitpush。因为 git pull 会把远程仓库的文件拉取到本地对应路径下，可能扰乱你原本的文件系统。用一个专门的空文件夹可以避免这个问题。

**第一步：在 GitHub 网页上创建一个新仓库（不在命令行操作）**

**第二步：进入你的项目文件夹**
```bash
cd 你的项目文件夹路径
```

**第三步：初始化本地 Git 仓库（只需一次）**
```bash
git init
```

**第四步：关联远程仓库地址（只需一次）**
```bash
git remote add origin https://github.com/你的用户名/仓库名.git
```

---

## 🔄 每次 Push 的流程

**第一步：进入项目文件夹**
```bash
cd 你的项目文件夹路径
```

**第二步：查看当前分支名（如果不确定是不是 main）**
```bash
git branch
```

**如果分支名不是 main，可以重命名（只需一次）**
```bash
git branch -m 原分支名 main
```

**第三步：拉取远程最新内容（如果有其他人更新过）**
```bash
git pull origin main
```

**第四步：查看哪些文件有改动（可选，但建议）**
```bash
git status
```

**第五步：添加文件**
```bash
git add .
# 如果只想添加指定文件：
git add 文件名
```

**第六步：提交到本地**
```bash
git commit -m "描述这次改了什么"
```

**第七步：推送到远程**
```bash
# 第一次 push 用这个（只需一次）：
git push -u origin main
# 之后每次直接用：
git push
```

---

## 📋 总结表格

| 步骤 | 命令 | 是否只需一次 |
|------|------|------|
| 初始化仓库 | `git init` | ✅ 只需一次 |
| 关联远程地址 | `git remote add origin ...` | ✅ 只需一次 |
| 设置上游分支 | `git push -u origin main` | ✅ 只需一次 |
| 查看/修改分支名 | `git branch` / `git branch -m` | 按需 |
| 拉取更新 | `git pull origin main` | 每次都需要 |
| 查看改动 | `git status` | 每次都需要 |
| 添加文件 | `git add .` | 每次都需要 |
| 提交 | `git commit -m "..."` | 每次都需要 |
| 推送 | `git push` | 每次都需要 |
