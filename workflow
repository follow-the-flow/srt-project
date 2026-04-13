# QArm学习进度日志
## 第七周：Lab01 Low Level Control

### 实验操作进展
- Part1，WristPosition，来回旋转手腕
- Part2，WristPWM，PWM Saturation和PWM Rate Limiter分别控制速度和加速度的上下限，超出部分截断
- Part2，Kp和Kd，可简单理解成前者正反馈，后者负反馈，是PD控制里的参数，详解见后
- Part3，WristCurrent，用Amplitude控制旋转角度的大小
- Part4，NonLinearEffects，Cmd Select不同，肩膀转动角度不同 （竖直面内前后转）

### 经验（熟悉simulink）
- Cmd Select选择接通下方的特定输入端口，发出预编译好的不同命令
- 可以直接在simulink里“电子断线”
- 读示波器：wrist angle里黄线为理想值，蓝线为实际值

### 概念和原理理解
- wrist controller模块里输入是wrist cmd（command）和wrist meas（measured），分别是设定值和当前值，传给PD控制，计算并用saturation和rate limitation限制，输出pwm（所以pwm直接控制电机转速，pwm的rate是电机的角加速度）
- PWM 本质上是一串只有高、低两种电平（0 或 1）的数字开关信号，本身不是连续电压。
    - 它通过在一个固定周期内，改变高电平所占的时间比例（即占空比），让电机在一个周期内得到等效的平均电压，从而用这种离散的开关方式，等效输出一个连续可调的模拟控制量。
- PD控制的输出是比例项（P）和微分项（D）的加权和，公式为：
    - $u(t) = K_p \cdot e(t) + K_d \cdot \frac{de(t)}{dt}$
    - $K_p$（比例增益）：对应图中的 Kp_wrist，作用是根据当前误差调整输出；误差越大，PWM 输出越强，让机械臂快速趋近目标。
    - $K_d$（微分增益）：对应图中的 Kd_wrist，作用是根据误差变化率抑制超调；当机械臂冲向目标时（误差快速减小），D 项反向输出，减速避免冲过头。
    - $e(t)$：位置误差，即 Wrist Cmd (目标角度) - Wrist Meas (实际角度)。

### 问题
- Part3，我们的QArm运作时加速度和速度有夸张的突变点
- pwm saturation和rate limitation在示波器的speed和acceleration里是绿线，按理说应该控制的挺死的，但为什么黄线有时会超过绿线的限制？（指导书里也是这样）
