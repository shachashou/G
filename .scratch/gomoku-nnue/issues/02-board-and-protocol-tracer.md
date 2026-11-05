# 02: 棋盘核心 + 协议最薄路径（随机合法走子引擎）

**What to build:** 打通全层最薄路径——引擎 exe 通过 stdin/stdout 说 Gomocup 协议子集（START / TURN / BOARD / INFO / ABOUT / END），内部维护 15×15 棋盘（落子/撤子/freestyle 胜负判定），收到 TURN 时回复一个随机但合法的着法。挂到现成 GUI 上，能与人类下完整一局（freestyle 语义，此时还未做禁手）。

**Blocked by:** 01（构建骨架与文档基线）

**Status:** ready-for-agent

- [ ] 棋盘 API：落子、撤子、查询占用、freestyle 胜负判定（任意方五连及以上获胜）、满盘和棋
- [ ] Gomocup 协议子集实现，能解析 BOARD 全量同步模式与 TURN 增量模式
- [ ] 对非法输入（越界、占位点、未知命令）有健壮处理，不崩溃
- [ ] exe 挂 GUI 或测试驱动脚本，用随机合法着法下完整一局并正确判胜负
- [ ] 协议黑盒测试：向进程写命令、断言输出格式
