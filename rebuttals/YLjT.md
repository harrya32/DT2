- (A) Decision-focused learning:

Discuss how the existing decision-focused learning (non-MDP) papers assume some ground truth, which we do not have for policy rankings, and we get around by using an OPE proxy (part of our contribution). This is similar for some of the MDP decision-focussed papers too, but some do use OPE to estimate policy value. But all have the 'task' of maximising the model for its optimal policy value, not ranking amongst a set of candidate policies, which is our unique setting. We are the first to port this literature stream with DTs, where this is the primary use case (human-in-the-loop decision making, not really for learning RL optimal policies). Then go into specific differences in each paper they cited to ours, and why it sholdn't be compared with. But acknowledge thanks for related work stream, it is indeed important to incorporate to place our work well and elucidate its contribution.

- (B) Comparison with offlime MBRL baselines. 

Report table below, comparing with MOReL, MOPO, and ROMI. Also describe how we already compared to an adapted version of VaGraM. We adapted it because VaGraM is an online MBRL method, and we also wanted to see whether encoding the pre-defined policy values in some way other than our ranking loss can benefit the model training (doesn't appear to much).


| Method | Pendulum | LunarLander | Hopper | Walker | Cheetah | Ant | Avg. Rank |
|----------|----------|----------|----------|----------|----------|----------|----------|
| DT2 |  0.03(0.02) / 0.720(0.064) | 0.62(0.62) / 0.960(0.023)  | 0.60(0.16) / 0.657(0.044)  | 0.21(0.08)/0.912(0.009)|  0.55(0.16)/0.896(0.023) | 1.64(0.68)/0.840(0.035) | 1.33 / 1.33 |
| NLL | 0.08(0.07) / 0.794 (0.106)| 2.96(1.65) / 0.909(0.027)  | 0.59(0.45) / 0.644(0.062)  | 0.55(0.30)/0.829(0.011)| 2.77(1.28)/0.481(0.164)| 7.10(1.56)/-0.086(0.155) | 3 / 3 |
| MOReL |  0.08(0.04) / 0.871(0.030) |   21.06(9.77) / 0.726(0.033)  |  0.51(0.17)/ 0.571(0.087)  | 0.35(0.06) / 0.874(0.019)|  5.08(1.20) / -0.286(0.119)  |  5.12(1.23) / 0.221(0.080)  | 2.67 / 3.17 |
| MOPO | 1.27(0.77)/0.417(0.080) | 1.45(0.52)/0.623(0.094)  | 8.50(4.19)/-0.274(0.123) | 8.13(2.31)/0.206(0.150)| 9.57(1.28)/-0.509(0.122) | 11.98(1.38)/0.051(0.131) | 5 / 5 |
| ROMI | 0.34()/0.404() |  8.50()/0.796() |0.64()/0.600()| 0.24()/0.829()| 13.76()/-0.829()| 8.59()/0.371()| 4.17 / 3.67 |
| HDTwin| 32.91(15.80)/-0.143(0.181)|49.10(39.01)/0.537(0.184)|0.94(0.49)/0.400(0.107)|7.50(1.15)/-0.131(0.159)|5.32(2.04)/0.157(0.225)|6.18(2.07)/0.243(0.153)| 4.83 / 4.83 | 