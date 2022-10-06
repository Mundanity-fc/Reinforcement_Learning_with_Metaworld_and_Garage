# An implementation of Metaworld with Garage

#### This Repo is a simple implementation of Multi-Task RL Experiments with [meta-world](https://github.com/rlworkgroup/metaworld) and [garage](https://github.com/rlworkgroup/garage/). Meta-world works as the environment and garage works as framework.

Partly followed [the document](https://garage.readthedocs.io/) of garage.

Due to the compiling error of mujoco, Windows Platform should install specific MuJoCo (Version 1.50) and mujoco-py(Version 1.50.1.68)

Note: During the installation of mujoco-py, pip may throw an error of 'the directory name is too long'. When same error occurs, try to copy the source code of mujoco-py directly into your Python libs folder.

This Repo has benn tested under Windows 11 (22H2) and Arch Linux (Kernel Zen-5.19.*)

---

#### 本仓库为多任务强化学习的一个简单实现，其中以 [meta-world](https://github.com/rlworkgroup/metaworld) 作为任务环境，并以 [garage](https://github.com/rlworkgroup/garage/) 作为训练框架。

过程中部分参考了 Garage 的[文档](https://garage.readthedocs.io/)

由于 mujoco 的编译问题，在 Windows 平台使用时，需要安装特定版本的 MuJoCo（版本 1.50）和 mujoco-py（版本 1.50.1.68）

注意：在安装 mujoco-py 的过程中，pip 可能会提示文件名或路径过长的报错，该错误可能由于 pip 过程中的缓存文件使用的哈希值导致路径过长而引起的。当遇到相同情况时，请直接将 mujoco-py 的源码复制进 Python 环境的 Libs 文件夹中

该仓库的内容均在 Windows 11（版本22H2）与 Arch Linux（内核Zen-5.19.*）下进行过测试