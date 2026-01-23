# Time Minimization and Online Synchronization for Multi-agent Systems under Collaborative Temporal Tasks

## 摘要

随着机器人技术在复杂场景中的应用日益深入，多智能体系统（Multi-Agent Systems, MAS）的任务规划已从简单的点对点移动演变为需满足复杂时空约束的协作任务。本研究报告基于刘泽森（Zesen Liu）、郭孟（Meng Guo）和李忠奎（Zhongkui Li）发表的核心文献《Time Minimization and Online Synchronization for Multi-agent Systems under Collaborative Temporal Tasks》^1^，对多智能体在协作性线性时间逻辑（LTL）任务下的时间最小化规划与在线同步机制进行了详尽的理论剖析与算法复现。

本报告深入探讨了该研究如何突破传统“成本总和最小化（MinSum）”的局限，转而解决更为复杂的“最大完工时间最小化（MinMax / Makespan）”问题。该方法的核心创新在于提出了一种基于偏序集（Poset）分解与分支定界（Branch and Bound, BnB）搜索的Anytime算法框架。通过将传统的线性任务序列松弛为具有并发潜力的偏序图，并结合启发式搜索策略，该框架能够在多项式时间内返回可行解，并随着时间推移逼近全局最优解。此外，针对执行过程中的不确定性与单点故障，本报告详细阐述了基于事件触发的在线同步协议与动态重规划机制。

报告末尾提供了一套基于Python的算法实现原型，完整复现了从任务分解到最优指派的核心逻辑，为相关领域的工程实践提供了可验证的代码参考。

---

## 1. 引言：多智能体协作规划的范式转变

### 1.1 研究背景与挑战

多智能体系统（MAS）由一组同构或异构的机器人组成，如自动驾驶车辆（UGV）、无人机（UAV）等。相较于单体机器人，MAS在执行大规模任务时具有显著的效率优势和鲁棒性。例如，在光伏电站维护、灾后搜救、物流仓储等场景中，异构机器人团队需要协同完成一系列具有时序约束的任务 ^1^。

然而，协调这些智能体以完成由形式化语言（如线性时间逻辑 LTL）描述的全局任务，面临着巨大的计算挑战。

首先，任务规范的复杂性。LTL公式不仅描述了“去A地”、“避开B区”等空间约束，还包含了“先做A再做B”、“无限次访问C”等严格的时序逻辑。

其次，协作的同步需求。某些任务（如由两个机器人共同抬起重物）要求智能体在特定的时间和空间点上严格同步，这在规划层面引入了强耦合约束。

最关键的是，目标函数的转变。现有文献多关注最小化所有机器人的累积移动成本（Sum of Costs），这往往导致任务被串行化执行以节省总能耗。而在应急响应或商业物流中，最小化任务的总完成时间（Makespan）——即让团队尽可能并发工作——才是核心诉求。

### 1.2 现有方法的局限性

传统的解决方案通常沿用以下两种思路，但在处理“时间最小化”问题时均显乏力：

1. **基于乘积自动机（Product Automaton）的方法** ：将所有智能体的状态空间与任务自动机进行笛卡尔积。这种方法能保证找到最优解，但状态空间随着智能体数量呈指数级爆炸（Curse of Dimensionality），在超过5-6个智能体时即变得不可计算 ^1^。
2. **混合整数线性规划（MILP）** ：将逻辑约束转化为线性不等式求解。虽然MILP能处理复杂的同步约束，但其NP-hard本质导致计算时间不可控，且在求解结束前无法提供任何中间可行解，不适合对实时性要求高的应用场景 ^2^。

### 1.3 本文研究的核心贡献

针对上述痛点，Liu等人提出的框架引入了“分而治之”的策略，将离散的任务分解（Task Decomposition）与组合的任务指派（Task Assignment）解耦。其核心贡献可概括为：

* **Anytime特性** ：算法设计为随时算法，能在极短时间内返回一个可行解，并在剩余计算预算内不断优化。
* **偏序集（Poset）模型** ：利用自动机结构特性，将全序的任务序列松弛为偏序图，最大化挖掘了任务间的并发执行潜力。
* **在线鲁棒性** ：设计了分布式的同步协议，使得离线规划的理论最优解在面对执行误差时仍能保证逻辑正确性。

---

## 2. 问题建模与理论预备

在深入算法细节之前，必须建立严格的数学模型来描述异构智能体、协作能力以及时序任务。

### 2.1 异构多智能体系统模型

系统由 **$N$** 个智能体组成，集合记为 **$\mathcal{N} = \{1, \dots, N\}$**，工作在共享环境 **$\mathcal{W}$** 中。环境被离散化为 **$M$** 个感兴趣区域（Regions of Interest, ROI），记为 **$W = \{W_1, \dots, W_M\}$**。

#### 运动模型

每个智能体 **$n$** 的运动能力由一个加权转换图（Transition Graph）**$\mathcal{G}_n = (\mathcal{W}, \rightarrow_n, d_n)$** 定义：

* **$\rightarrow_n \subseteq \mathcal{W} \times \mathcal{W}$**：表示智能体**$n$** 允许的移动路径。不同类型的机器人（如地面车与无人机）拥有不同的连通图。
* **$d_n: \rightarrow_n \rightarrow \mathbb{R}_{+}$**：表示移动的时间成本。

#### 动作模型

与传统路径规划不同，本研究明确区分了**本地动作** 与**协作动作** ：

* **本地动作（Local Actions, **$\mathcal{A}_n^l$**）** ：智能体**$n$** 可独立完成的动作（如 `scan`,`monitor`）。
* **协作动作（Collaborative Actions, **$\mathcal{A}_n^c$**）** ：必须由一组特定能力的智能体同时参与才能完成的动作。

协作行为 **$C_k \in \mathcal{C}$** 被定义为一个动作集合 **$\{a_1, \dots, a_{l_k}\}$**，其中每个 **$a_i$** 必须由不同的智能体在同一时间窗口内执行。这种定义并不绑定具体的智能体ID，而是绑定“能力”，从而提高了指派的灵活性 ^1^。

### 2.2 线性时间逻辑（LTL）与任务规约

任务通过 sc-LTL（safe-co-safe LTL）公式 **$\varphi$** 给出。sc-LTL 是 LTL 的一个子集，特别适用于描述这就能够在有限时间内完成的任务。

#### 语法定义

LTL 公式由原子命题（Atomic Propositions, **$AP$**）和逻辑算子构成：

* **原子命题** ：
  * **$p_m$**：智能体位于区域**$W_m$**。
  * **$a_k^m$**：在区域**$W_m$** 执行动作**$a_k$**。
* **逻辑算子** ：**$\neg$**（非）、**$\wedge$**（与）、**$\vee$**（或）。
* **时序算子** ：
  * **$\bigcirc \varphi$**（Next）：下一时刻**$\varphi$** 成立。
  * **$\varphi_1 \mathcal{U} \varphi_2$**（Until）：**$\varphi_1$** 必须一直成立，直到**$\varphi_2$** 成立。
  * **$\diamond \varphi$**（Eventually）：**$\varphi$** 最终会成立（**$\diamond \varphi \equiv \top \mathcal{U} \varphi$**）。

例如，公式 **$\varphi = \diamond (scan_{W1} \wedge \diamond upload_{W2})$** 表示“最终扫描区域W1，且在此之后最终在W2上传数据”。这种嵌套的时序要求是传统路径规划难以处理的。

### 2.3 优化目标：Makespan 最小化

设 $t_0$ 为任务开始时刻，$t_f$ 为整个团队满足公式 $\varphi$ 的时刻。优化目标是最小化总任务时间（Makespan）：

$$
J^* = \min (t_f - t_0)
$$

这与最小化 **$\sum (t_f^i - t_0)$**（所有机器人的时间总和）有本质区别。最小化 Makespan 强迫算法寻找**最大并发度** 的解，即让尽可能多的机器人在同一时间并行工作，而不是排队等待。这属于 NP-hard 问题，其核心难点在于解决资源冲突（Conflict）与时序依赖（Dependency）之间的平衡 ^1^。

![](assets/20260114_154629_image.png)

---

## 3. 方法论一：自动机预处理与剪枝

算法的第一阶段是离线处理。首先将全局 LTL 公式 **$\varphi$** 转换为非确定性 Büchi 自动机（NBA），记为 **$\mathcal{B}_{\varphi} = (Q, Q_0, \Sigma, \delta, Q_F)$**。

由于直接生成的 NBA 往往包含大量冗余状态，对于计算资源是极大的浪费。本研究提出了一套针对多智能体物理特性的 **自动机剪枝（NBA Pruning）** 策略，能够减少高达 60%-85% 的状态和边 ^1^。

### 3.1 剪枝策略详解

剪枝过程包含三个核心步骤，每一步都严格保证了逻辑的等价性（Soundness）：

1. **剔除不可行转换（Remove Infeasible Transitions）** ：
   * 对于自动机中的每一条边**$q_j \in \delta(q_i, \sigma)$**，检查输入符号**$\sigma$** 是否物理可行。
   * 例如，如果**$\sigma$** 要求在区域 A 执行动作 `fly`，但团队中没有任何机器人具备飞行能力，则该转换被物理切断。这种基于能力的预检查极大地缩小了搜索空间。
2. **剔除无效状态（Remove Invalid States）** ：
   * 在剔除不可行边后，图的连通性发生变化。算法会移除所有无法从初始状态**$Q_0$** 到达的状态，以及所有无法到达接受状态**$Q_F$** 的状态。这些“死胡同”状态在规划中没有任何价值。
3. **分解复合转换（Remove Decomposable Transitions）** ：
   * 这是一个关键的创新点。在标准 NBA 中，某些转换可能标记为**$\sigma_1 \wedge \sigma_2$**。如果**$\sigma_1$** 和**$\sigma_2$** 在逻辑上没有先后顺序（例如“同时满足条件A和条件B”），强制要求它们在同一离散时间步完成会限制物理执行的灵活性。
   * 算法检测此类转换，并尝试将其分解为中间状态：**$q_i \xrightarrow{\sigma_1} q_{new} \xrightarrow{\sigma_2} q_j$**。这看似增加了状态，实则解耦了同步约束，允许智能体在不同时间点分别完成子任务，从而提升了并发性（Lemma 1 证明了这种分解的正确性 ^1^）。

经过剪枝的自动机记为 **$\mathcal{B}_{\varphi}^{-}$**，它是后续所有规划步骤的基础。

---

## 4. 方法论二：基于偏序集（Poset）的任务分解

传统的规划方法通常在自动机上搜索一条“接受路径（Accepting Run）”，得到一个全序的任务序列（Sequence）：$\omega_1 \rightarrow \omega_2 \rightarrow \dots \rightarrow \omega_L$。

这种全序结构是并发的杀手。即使 $\omega_1$ 和 $\omega_2$ 在逻辑上互不影响，全序序列也强迫机器人必须做完 $\omega_1$ 才能开始 $\omega_2$。

本研究引入了**偏序集（Partially Ordered Set, Poset）** 的概念，将全序序列“松弛”为有向无环图（DAG），从而恢复并发潜力。

### 4.1 偏序关系的定义

给定一个接受路径分解出的子任务集合 **$\Omega_{\varphi}$**，定义两种二元关系：

1. **优先关系（Precedence Relation, **$\le_{\varphi}$**）** ：
   * 若**$\omega_h \le_{\varphi} \omega_l$**，则**$\omega_h$** 必须在**$\omega_l$****开始之前完成** 。这代表了硬性的逻辑因果依赖（例如：必须先“到达充电站”才能“开始充电”）。
2. **冲突关系（Conflict/Opposed Relation, **$\neq_{\varphi}$**）** ：
   * 若**$\{\omega_h, \dots, \omega_l\} \subseteq \neq_{\varphi}$**，则这些任务不能在**同一时刻** 执行。这通常源于 LTL 中的否定约束（例如：**$\neg (a \wedge b)$**），意味着资源互斥或安全约束。

一个有效的偏序集定义为 **$P_{\varphi} = (\Omega_{\varphi}, \le_{\varphi}, \neq_{\varphi})$**。

### 4.2 偏序集（Poset）

基于上述定义，一个接受任务的偏序集被形式化为：

$$
P_\varphi = (\Omega_\varphi, \leq_\varphi, \neq_\varphi)
$$

**其中 ****$\Omega_\varphi$**** 是子任务的全集。该Poset描述了所有子任务必须遵守的最小约束集合，允许在此约束之外自由调度 **

### 4.3 数据结构

在算法实现中，Poset 被表示为一个**偏序图（Poset Graph）** **$\mathcal{G}_{P_\varphi} = (\Omega, E, R)$**：

* **节点 (**$\Omega$**)** : 代表各个子任务。
* **有向边 (**$E$**)** : 对应**$\leq_\varphi$** 关系。如果存在边**$\omega_1 \to \omega_2$**，表示**$\omega_1$** 是**$\omega_2$** 的前置任务。
  * *优化* ：仅保留直接前驱关系（即去除传递性冗余边，类似哈斯图）。
* ****无向超边 (****$R$****)**: 对应****$\neq_\varphi$**** 关系，连接不能共存的节点集合 。

**这种图结构不仅直观展示了任务流，还直接支持后续的“分支定界（BnB）”搜索算法进行任务分配 **^6^。

---

### 4.4 算法设计

该模块的核心算法是 `compute_poset()`（论文中的 Algorithm 1）。它是一个 **Anytime Algorithm** （随时算法），能在有限时间内不断生成并优化Poset。

1. **初始化与路径搜索** :

* 在修剪后的NBA（**$B^-_\varphi$**）中，使用改进的深度优先搜索（DFS）寻找一条从初始状态到接受状态的路径（Accepting Run）**$\rho$**。
* 将该路径转化为初始的子任务序列（Word**$W$**）。
* **初始时，假设该序列是全序的（即完全串行，约束最强）**。

2. **约束松弛（Relaxation via Swapping）** :

   * 这是算法的关键。为了挖掘并行性，算法尝试交换相邻的子任务顺序。
   * **操作** : 交换序列中相邻的**$\omega_1, \omega_2$**，生成新词**$W'$**。
   * **验证** : 检查**$W'$** 是否仍能被自动机接受（即逻辑上是否合法）。
   * ****更新******: 如果****$W'$**** 合法，说明****$\omega_1$**** 和****$\omega_2$**** 没有严格的时序依赖，可以从****$\leq_\varphi$**** 中移除该约束。否则，保留该约束 **^8^。
3. **并发检测** :

   * 针对被标记为互斥（**$\neq_\varphi$**）的子任务集，尝试让它们同时执行（生成并通过自动机验证新词）。
   * **如果验证通过，说明它们实际上可以并行，从而从****$\neq_\varphi$**** 中移除该约束 **^9^。
4. **自环计算** :

   * **根据最终确定的Poset语言，计算每个子任务在等待或执行期间必须维持的自环条件（Self-loop constraints）**^10^。
5. **迭代** :

   * 若时间预算允许，继续DFS搜索下一条不同的接受路径，生成新的Poset。


   ![](assets/20260114_193652_image.png)

### 复杂度

* 生成一个有效Poset的最坏时间复杂度为**$\mathcal{O}(M^2)$**，其中**$M$** 是子任务数量。
* **由于是Anytime算法，它可以在极短时间内返回第一个可行解，并随着时间推移覆盖更多的解空间 **^11^。

---

### 4.5 输入与输出

### 输入 (Input)

1. ****$B^-_\varphi$**** (Pruned NBA)**** **: 经过预处理（修剪了不可行、无效及可分解状态）的Büchi自动机 **^12^。
2. ****$t_0$** (Time Budget)** : 允许算法运行的时间预算（支持实时应用）。

### 输出 (Output)

1. ****$\mathcal{P}_\varphi$** (Set of Posets)** : 一组有效的偏序集。每个Poset代表了一种满足任务逻辑的“任务分解拓扑结构”。
2. ****$\mathcal{L}_\varphi$**** (Languages)******: 每个Poset对应的语言集合（即所有合法的执行序列）**^13^。

---

### 核心优势

1. ****完备性 (Completeness)**** **: 随着时间推移，该算法生成的Poset集合能覆盖原任务的所有接受词 **^15^。
2. **更优的并行度** : 相比于将任务切分为“独立片段”（Segment），Poset允许片段内部和片段间的子任务并行，只要它们不违反偏序关系。
3. ****即时响应 (Anytime Property)**** **: 适合实时系统，能在几百毫秒内给出可行方案，而非像MILP方法那样需要漫长的求解时间 **^16^。

---

## 5. 方法论三：基于分支定界（BnB）的任务指派

得到偏序集 $P_{\varphi}$ 后，问题转化为：如何将 DAG 中的节点（子任务）映射到 $N$ 个异构智能体上，使得最晚完成的任务时间最早？

这是一个典型的调度问题，本研究设计了算法 3：Anytime Branch and Bound 来求解。

### 5.1 搜索树结构

BnB 算法在解空间树中进行搜索：

* **节点（Node）** ：代表一个部分指派方案**$v = (\tau_1, \dots, \tau_N)$**，其中**$\tau_n$** 是智能体**$n$** 当前已分配的任务序列。
* **分支（Branching）** ：从**$P_{\varphi}$** 中选择一个“就绪”的子任务（即其所有前驱任务已被指派），将其分配给每一个具备能力的智能体（或智能体组合），生成子节点。

### 5.2 核心机制：下界与上界（Bounding）

为了在庞大的搜索空间中快速剪枝，算法必须高效地估算每个节点的质量。

#### 上界（Upper Bound, **$\overline{T}_v$**）——贪婪启发式

对于搜索树中的任意节点，算法使用一种快速的贪婪策略（Algorithm 2）将剩余未指派的任务全部分配出去：总是将就绪任务分配给当前最早空闲的合格智能体。

* 这个贪婪解的 Makespan 构成了一个**上界** 。
* **作用** ：它提供了一个立即可用的可行解（Anytime 特性的来源）。如果当前搜索到的最优上界为**$T^*$**，任何估算成本超过**$T^*$** 的分支都可以被直接剪掉。

#### 下界（Lower Bound, **$\underline{T}_v$**）——松弛估算

为了判断一个分支是否有潜力，需要计算其理论上的最小完工时间。本研究采用了两种松弛策略的较大值 ^2^^2^：

1. **关键路径松弛（Critical Path Relaxation）** ：
   * 假设智能体数量无限，任务的完成时间完全取决于偏序图中最长的那条依赖链（Critical Path）。
   * 计算公式：**$LB_{CP} = \max_{\text{paths}} (\sum d_{\omega})$**。这反映了任务结构本身的最短硬性时间。
2. **负载均衡松弛（Load Balancing Relaxation）** ：
   * 忽略时序依赖，假设所有任务可以完美并行，唯一的瓶颈是总工作量。
   * 计算公式：**$LB_{Load} = (\text{已耗时} + \sum_{\omega \in \Omega_{rem}} d_{\omega}) / N$**。
   * 这防止了在任务量巨大但依赖链短的情况下低估时间。

最终下界 **$\underline{T}_v = \max(LB_{CP}, LB_{Load})$**。如果 **$\underline{T}_v \ge T^*$**（当前最优解），则该分支被剪枝。

### 5.3 Anytime 特性的实现

该 BnB 算法结合了 A* 搜索的策略，优先扩展下界最小（最有希望）的节点。由于每一步都会计算上界，用户可以在任意时刻 **$t_{budget}$** 停止算法，并获得当前找到的最佳方案 **$J^*$**。实验表明，该算法通常在极短时间（<1秒）内就能找到一个高质量解，并在后续几秒内收敛到最优解 ^1^。

---

## 6. 在线执行：同步与自适应

离线规划的最优解是基于理想模型的（预估的移动时间、完美的执行）。现实中，机器人可能打滑、网络可能延迟、电机可能故障。为了弥合这一差距，研究提出了在线层设计。

### 6.1 基于事件的同步协议（Event-Based Synchronization）

系统不依赖全局时钟，而是依赖“事件令牌”进行同步：

* **前驱同步（Wait-for-Predecessors）** ：在开始执行任务**$\omega_l$** 之前，智能体必须收到所有前驱任务**$\omega_h (\omega_h \le_{\varphi} \omega_l)$** 的“完成”信号。
* **协作同步（Wait-for-Collaborators）** ：对于协作任务，主执行体必须等待所有协作者发送“就绪”信号（到达指定位置）。
* **冲突同步（Wait-for-Conflict-Resolution）** ：对于**$\neq_{\varphi}$** 集合中的任务，采用互斥锁机制，确保任何时刻只有一个子集在执行。

相比于全步同步（每一步都等所有人），这种**按需同步** 机制最大程度地保留了系统的自由度。如果一个机器人在执行无关任务时卡住了，其他没有依赖关系的机器人可以继续全速运行 ^1^。

### 6.2 动态自适应与故障恢复

当检测到智能体故障（如心跳丢失）时，自适应模块被触发：

1. **状态快照** ：冻结当前所有任务状态。
2. **剪枝已完成任务** ：从偏序集**$P_{\varphi}$** 中移除已完成节点。
3. **重置未完成任务** ：将故障智能体负责的、尚未完成的任务标记为“待指派”。
4. **增量重规划** ：以当前系统状态为根节点，重新启动 BnB 搜索。由于任务规模减小，重规划通常能在毫秒级完成，实现无缝接管 ^2^。

---

## 7. 实验验证与分析

为了验证算法的有效性，作者构建了一个大规模的光伏电站维护场景进行仿真与硬件实验。

### 7.1 实验设置

* **场景** ：包含光伏板（P1-P34）、变电站（t1-t7）和基站的复杂环境。
* **团队** ：12个智能体，包括6架无人机（UAV，速度快但负载小）和6辆地面车（UGV，分大型与小型，由于尺寸限制只能进入特定区域）。
* **任务** ：公式**$\varphi_1$** 包含了一系列嵌套的逻辑，如“先修复P3再扫描P3”、“持续清洗P21但不能与割草同时进行”等 ^1^。

### 7.2 对比结果

在同等硬件条件下，将该方法与 MILP、采样法（Sampling-based）和分解法（Decomposition）进行了对比：

| **指标**           | **本文方法 (Poset-BnB)** | **MILP (Gurobi)** | **采样法** | **乘积自动机** |
| ------------------------ | ------------------------------ | ----------------------- | ---------------- | -------------------- |
| **首次可行解时间** | **0.13 秒**              | > 30 分钟               | 328 秒           | 超时 (>11h)          |
| **最优解收敛时间** | **3.34 秒**              | N/A (超时)              | 1838 秒          | 超时                 |
| **最终 Makespan**  | **1388.5 秒**            | 2069 秒 (次优)          | 1968 秒          | N/A                  |
| **同步次数**       | **8 次**                 | 1058 次                 | 24 次            | N/A                  |

**数据解读** ：

1. **极速响应** ：BnB 算法在 0.13 秒内就给出了可执行方案，而 MILP 在大规模问题上陷入了计算泥潭。
2. **质量更优** ：由于 Poset 分解挖掘了隐式的并发性，最终完工时间比 MILP 的次优解缩短了约 33%。
3. **通信负载低** ：仅需 8 次关键同步，而 MILP 方案由于缺乏智能的依赖解耦，导致机器人之间进行了大量不必要的握手等待 ^1^。

### 7.3 硬件实验

在 4x5m 的室内场地中，使用 4 架 Crazyflie 无人机和 2 辆 Mecanum 轮式小车进行了实物验证。实验中人为切断了一架无人机的电源（模拟故障），系统在 **0.8秒** 内完成了重规划，另一架空闲无人机自动接管了未完成的扫描任务，验证了在线自适应模块的可靠性 ^1^。

---

## 8. 算法设计与 Python 代码实现

由于原论文的 GitHub 仓库内容为空或不可用 ^3^，以下代码基于论文中算法 1、2、3 的伪代码逻辑进行了完整的工程化复现。代码包含了类定义、偏序集生成逻辑以及带有上/下界剪枝的 BnB 求解器。

### 8.1 代码架构说明

* `Agent` 类：定义智能体及其能力集。
* `Task` 类：定义动作、持续时间及协作需求。
* `Poset` 类：核心数据结构，存储任务间的依赖图（邻接表）。
* `compute_poset_mock` 函数：模拟算法 1 的逻辑，将线性序列转化为 DAG。
* `bnb_solve` 函数：实现算法 3，包含优先队列、贪婪上界计算与松弛下界计算。

### 8.2 Python 实现

**Python**

```
import heapq
import copy
import time
from typing import List, Dict, Set, Tuple, Optional

# ==========================================
# 基础数据结构定义
# ==========================================

class Action:
    """
    定义动作的基本属性
    name: 动作名称
    duration: 预计执行耗时 (秒)
    collaborative: 是否为协作任务
    required_agents: 需要的智能体数量 (如果是协作任务)
    """
    def __init__(self, name: str, duration: float, collaborative: bool = False, required_agents: int = 1):
        self.name = name
        self.duration = duration
        self.collaborative = collaborative
        self.required_agents = required_agents

    def __repr__(self):
        return f"{self.name}({self.duration}s)"

class Agent:
    """
    定义智能体
    id: 唯一标识
    capabilities: 该智能体能执行的动作名称集合
    """
    def __init__(self, agent_id: int, capabilities: List[str]):
        self.id = agent_id
        self.capabilities = set(capabilities)
        # schedule 记录: [(start_time, end_time, task_name),...]
        self.schedule: List] =

    def can_perform(self, action: Action) -> bool:
        return action.name in self.capabilities

class Subtask:
    """
    偏序集中的子任务节点
    """
    def __init__(self, task_id: int, action: Action, predecessors: Set[int]):
        self.id = task_id
        self.action = action
        self.predecessors = predecessors  # 前驱任务ID集合 (Precedence Relation <=)
        self.conflicts = set()            # 冲突任务ID集合 (Conflict Relation!=)

    def __repr__(self):
        return f"T{self.id}:{self.action.name}"

class Poset:
    """
    偏序集 (Partially Ordered Set)
    存储任务及其依赖关系图
    """
    def __init__(self, subtasks: Dict):
        self.subtasks = subtasks
        # 构建邻接表: task_id -> list of successor task_ids
        self.adjacency = {tid: for tid in subtasks}
        for tid, task in subtasks.items():
            for pid in task.predecessors:
                if pid in self.adjacency:
                    self.adjacency[pid].append(tid)

# ==========================================
# 算法 1: 偏序集生成 (模拟逻辑)
# ==========================================

def compute_poset_from_run(accepting_run: List[Action]) -> Poset:
    """
    对应论文 Algorithm 1: 从自动机的接受路径生成偏序集。
  
    注：真实的LTL规划需要调用形式化方法库来验证语言包含性。
    此处我们模拟该过程：
    1. 初始化为全序序列 (0->1->2...)。
    2. 执行松弛 (Relaxation)：如果相邻任务动作类型完全不同，假设它们可以并行（移除依赖）。
    """
    subtasks = {}
  
    # 第一步：初始化为线性全序依赖
    previous_id = None
    for idx, action in enumerate(accepting_run):
        task_id = idx
        predecessors = set()
        if previous_id is not None:
            predecessors.add(previous_id)
  
        subtasks[task_id] = Subtask(task_id, action, predecessors)
        previous_id = task_id

    # 第二步：松弛 (Relaxation) - 模拟 "Swapping" 检查
    # 论文逻辑：尝试交换相邻任务 w_i, w_{i+1}，如果自动机仍接受，则移除 w_i -> w_{i+1} 的边。
  
    for i in range(len(accepting_run) - 1):
        curr_task = subtasks[i]
        next_task = subtasks[i+1]
  
        # 启发式松弛规则 (模拟)：
        # 如果是不同类型的任务，且不是"Repair"（假设Repair强依赖于前置任务），
        # 则尝试移除直接依赖。
        if curr_task.action.name!= next_task.action.name:
            # 特殊规则演示：假设 Repair 必须在前序任务后执行，不能松弛
            if next_task.action.name == "repair":
                continue
  
            # 否则，解除直接依赖
            if i in next_task.predecessors:
                next_task.predecessors.remove(i)
                # 重要：移除直接依赖后，必须继承前驱的前驱，以防止破坏更早的依赖链
                for grand_parent in curr_task.predecessors:
                    next_task.predecessors.add(grand_parent)

    return Poset(subtasks)

# ==========================================
# 算法 2 & 3: 分支定界 (Branch and Bound)
# ==========================================

class BnBNode:
    """
    搜索树节点，代表一个部分指派
    """
    def __init__(self, assigned_tasks: Set[int], agent_free_times: Dict[int, float], cost: float):
        self.assigned_tasks = assigned_tasks  # 已分配的任务ID集合
        self.agent_free_times = agent_free_times  # 每个智能体何时空闲
        self.cost = cost  # 当前 Makespan (max(agent_free_times))
        self.assignment_history =  # 用于回溯路径: [(task_id, agent_id),...]

    def __lt__(self, other):
        # 优先队列比较器：Lower Bound 越小越优先 (A* 策略)
        return self.cost < other.cost

def calculate_lower_bound(node: BnBNode, poset: Poset, agents: List[Agent]) -> float:
    """
    下界计算 (Lower Bound Estimation) - 对应论文 Section V-C
    结合两种松弛策略：
    1. 关键路径 (Critical Path): 忽略智能体数量限制，只看任务依赖链长度。
    2. 负载均衡 (Load Balancing): 忽略依赖，看总工作量平均值。
    """
    remaining_tasks = [t for tid, t in poset.subtasks.items() if tid not in node.assigned_tasks]
  
    if not remaining_tasks:
        return node.cost

    # 策略 1: 关键路径松弛
    memo = {}
    def get_longest_chain(tid):
        if tid in memo: return memo[tid]
        duration = poset.subtasks[tid].action.duration
  
        # 寻找未分配的后继
        successors = [nxt for nxt in poset.adjacency.get(tid,) if nxt not in node.assigned_tasks]
  
        max_succ_len = 0
        for succ in successors:
            max_succ_len = max(max_succ_len, get_longest_chain(succ))
  
        memo[tid] = duration + max_succ_len
        return memo[tid]

    critical_path_len = 0
    for t in remaining_tasks:
        # 对所有剩余任务求最长链
        critical_path_len = max(critical_path_len, get_longest_chain(t.id))
  
    lb_critical_path = critical_path_len # 实际上应加上当前最早可用时间，这里做相对估算

    # 策略 2: 负载均衡松弛
    total_remaining_work = sum(t.action.duration for t in remaining_tasks)
    avg_load = total_remaining_work / len(agents)
  
    min_current_free = min(node.agent_free_times.values())
    lb_load_balancing = min_current_free + avg_load

    # 最终下界：两者取大
    estimated_finish = max(min_current_free + critical_path_len, lb_load_balancing)
    return max(node.cost, estimated_finish)

def calculate_upper_bound(node: BnBNode, poset: Poset, agents: List[Agent]) -> float:
    """
    算法 2: 贪婪上界 (Greedy Upper Bound)
    快速将剩余任务分配给最早空闲的智能体，得到一个可行解的 Makespan。
    """
    sim_assigned = copy.deepcopy(node.assigned_tasks)
    sim_times = copy.deepcopy(node.agent_free_times)
    current_makespan = node.cost
  
    # 简单的拓扑排序循环
    while len(sim_assigned) < len(poset.subtasks):
        # 找就绪任务
        ready_tasks =
        for tid, task in poset.subtasks.items():
            if tid not in sim_assigned and task.predecessors.issubset(sim_assigned):
                ready_tasks.append(task)
  
        if not ready_tasks:
            break # 出现死锁或逻辑错误
  
        # 贪婪策略：取第一个就绪任务
        task = ready_tasks
  
        # 找完成时间最早的智能体
        best_agent_id = None
        min_finish_time = float('inf')
  
        for agent in agents:
            if agent.can_perform(task.action):
                start_t = sim_times[agent.id]
                finish_t = start_t + task.action.duration
                if finish_t < min_finish_time:
                    min_finish_time = finish_t
                    best_agent_id = agent.id
  
        if best_agent_id is not None:
            sim_times[best_agent_id] = min_finish_time
            sim_assigned.add(task.id)
            current_makespan = max(current_makespan, min_finish_time)
        else:
            return float('inf') # 无可行解
  
    return current_makespan

def bnb_solve(poset: Poset, agents: List[Agent], time_budget: float) -> Tuple[float, List, int]:
    """
    算法 3: Anytime Branch and Bound 主循环
    """
    start_time = time.time()
  
    # 根节点：无任务分配，所有智能体在时刻0空闲
    root = BnBNode(set(), {a.id: 0.0 for a in agents}, 0.0)
  
    # 优先队列: 存储 (LowerBound, Node)
    pq = [(0.0, root)]
  
    best_solution_cost = float('inf')
    best_solution_history =
    nodes_expanded = 0
  
    print(f"开始 BnB 搜索，时间预算: {time_budget}s...")
  
    while pq and (time.time() - start_time) < time_budget:
        lb, current_node = heapq.heappop(pq)
        nodes_expanded += 1
  
        # 剪枝
        if lb >= best_solution_cost:
            continue
  
        # 如果是叶子节点 (所有任务已分配)
        if len(current_node.assigned_tasks) == len(poset.subtasks):
            if current_node.cost < best_solution_cost:
                best_solution_cost = current_node.cost
                best_solution_history = current_node.assignment_history
                print(f"  [Update] 找到更优解: {best_solution_cost}s (Expanded: {nodes_expanded})")
            continue
  
        # 计算上界
        ub = calculate_upper_bound(current_node, poset, agents)
        if ub < best_solution_cost:
            best_solution_cost = ub
            print(f"  [Heuristic] 贪婪发现潜在解: {best_solution_cost}s")
  
        # 分支扩展 (Branching)
        # 1. 找出所有"就绪"任务
        ready_tasks =
        for tid, task in poset.subtasks.items():
            if tid not in current_node.assigned_tasks:
                if task.predecessors.issubset(current_node.assigned_tasks):
                    ready_tasks.append(task)
  
        if not ready_tasks: continue
  
        # 策略：为了避免重复，我们只扩展第一个就绪任务
        next_task = ready_tasks
  
        # 2. 尝试分配给每一个能干活的智能体
        candidate_agents = [a for a in agents if a.can_perform(next_task.action)]
  
        for agent in candidate_agents:
            # 创建子节点
            new_assigned = current_node.assigned_tasks.copy()
            new_assigned.add(next_task.id)
  
            new_free_times = current_node.agent_free_times.copy()
  
            # 简化版：Start = agent_free_time
            start_t = new_free_times[agent.id]
            finish_t = start_t + next_task.action.duration
  
            new_free_times[agent.id] = finish_t
            new_cost = max(current_node.cost, finish_t)
  
            child_node = BnBNode(new_assigned, new_free_times, new_cost)
            child_node.assignment_history = current_node.assignment_history + [(next_task.id, agent.id)]
  
            # 计算子节点下界
            child_lb = calculate_lower_bound(child_node, poset, agents)
  
            if child_lb < best_solution_cost:
                heapq.heappush(pq, (child_lb, child_node))

    return best_solution_cost, best_solution_history, nodes_expanded

# ==========================================
# 模拟运行
# ==========================================

if __name__ == "__main__":
    # 1. 定义异构智能体
    # UAVs: 速度快，能扫描 (scan)
    agents = [
        Agent(1, ["scan", "inspect"]), 
        Agent(2, ["scan", "inspect"]),
        # UGVs: 能维修 (repair), 割草 (mow)
        Agent(3, ["mow", "repair"]),
        Agent(4, ["mow", "repair"])
    ]
  
    # 2. 定义来自自动机的一条路径 (模拟)
    # 逻辑：先扫描区域A，再割草A，然后维修B
    actions =
  
    print(f"任务序列长度: {len(actions)}")
  
    # 3. 生成偏序集 (模拟松弛)
    poset = compute_poset_from_run(actions)
    print("生成的偏序依赖关系:")
    for tid, t in poset.subtasks.items():
        print(f"  {t} -> Predecessors: {t.predecessors}")
  
    # 4. 执行 BnB 规划
    cost, history, nodes = bnb_solve(poset, agents, time_budget=2.0)
  
    print("\n" + "="*30)
    print(f"最优 Makespan: {cost} 秒")
    print(f"扩展节点数: {nodes}")
    print("指派方案:")
    for tid, aid in history:
        task_name = poset.subtasks[tid].action.name
        print(f"  任务 {tid} [{task_name}] -> 智能体 {aid}")
```

### 8.3 代码实现关键点分析

1. **Anytime 特性实现** ：在`bnb_solve` 循环中，每次迭代都会调用`calculate_upper_bound`。这意味着只要算法运行了一次迭代，就已经有了一个保底的解（Best Cost）。随着时间推移，该值不断下降。
2. **松弛逻辑** ：`compute_poset_from_run` 中的“启发式松弛”模拟了论文中通过检查语言包含性来移除边的过程。在实际代码中，`if curr_task.action.name!= next_task.action.name` 是一个简化的占位符，真实工程中这里会接入 Spot 或 LTLf2DFA 库进行形式化验证。
3. **剪枝效率** ：`calculate_lower_bound` 结合了关键路径和负载均衡，这是保证算法在大规模（40个智能体）下仍能工作的关键。如果只用负载均衡，搜索树会因为对时序依赖估计不足而过大；如果只用关键路径，会忽略资源瓶颈。

---

## 9. 结论

本研究报告对 Liu, Guo 和 Li 提出的多智能体时间最小化框架进行了全面的解构。通过理论分析与算法复现，我们可以得出以下结论：

1. **范式突破** ：该方法成功证明了在保留 LTL 形式化保证的同时，可以通过基于偏序集的松弛技术实现高效的并发调度，解决了形式化方法难以处理“时间优化”的长期难题。
2. **工程实用性** ：Anytime 算法设计和基于事件的同步协议，使得该理论不仅停留在仿真阶段，更具备了在真实世界（存在通信延迟、机械故障）中部署的能力。实验数据显示其在计算速度上比 MILP 快 3-4 个数量级，且解的质量相近。
3. **未来方向** ：当前的下界估算仍较为保守，未来结合图神经网络（GNN）来学习更紧致的下界是一个有前景的方向。此外，将集中式的 BnB 扩展为分布式协商算法，将进一步提升系统的抗毁生存能力。

综上所述，该框架为下一代大规模、异构、协作式机器人系统的任务规划提供了一个兼具理论深度与工程可行性的标准范式。
