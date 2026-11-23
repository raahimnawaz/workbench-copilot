# Design notes and results

Each idea, the research it lines up with, and what was measured; then the timings from the demo takes and what ran live. Back to the [README](../README.md).

## Design notes

What each idea is, the research it lines up with, and what was measured. The papers are under [References](#references).

- **The Toolformer-style fine-tune** (`scripts/toolformer_*.py`): adapted from Meta's Toolformer [1], which teaches a model a tool from labels written for it, keeping only the calls that help. Here Spark writes spoken requests and labels each in the fast path's own prompt, and a label is kept only when it matches the intent it was written for and its call runs on the recorded bench, where Toolformer keeps a call when it lowers the model's loss. The student is Llama 3.2 3B [3] with a LoRA [13]. Measured: 351 of 390 labels kept, for $0.99; trained on Modal, the right tool on held-out requests went from 47 to 50 of 53. [The fast path's section](agent.md#the-arms-fast-path-rules-llama-and-a-toolformer-style-fine-tune) has the whole pipeline.
- **Reading the board with SAM:** SAM 3.1 [2] segments the wires from one noun phrase, and plain code does the rest: an AprilTag carrier [14] maps the photo to millimeters, so there's no camera calibration to keep, and the hole grid is fitted to the photo, searching 1.2 mm, 1° and a scale of 0.97–1.06 around the printed outline; a wire end snaps to a hole within 1.3 mm. Measured: live on the bench, it finds the demo's wire on the wrong GPIO with the line to fix.
- **The arm's fast path:** a cheap local model first and Spark as the fallback, an LLM cascade like FrugalGPT [7] and RouteLLM [8]. What's different: the arm moves only when two independent checks agree, plain rules and Llama 3.2 3B, and anything else goes to Spark. Measured: about 0.25 s from request to reply, against about 4.4 s through Spark; on 117 phrasings neither was tuned on, no wrong arm move, where the model alone made 20 and the rules alone 3.
- **Hiding dead time:** bench facts fetched before Spark's first call and kept for the session, tool calls in one round run in parallel, and a request that asks for an arm move and something else runs both at once, so Spark Open can work while the arm moves. The same idea as LLMCompiler's parallel function calling [9]. Measured: a follow-up Spark call went from 12–14 s to about 2 s, and the wait on models fell by about 90% over the build.
- **Memory recall:** rare-word weighting and a slow fade with age, the same as IDF [10] and BM25 [11] and DCG's log discount [12]; turns that changed the bench count more, as Generative Agents weighs importance beside recency and relevance [6]. So "where does my screwdriver go" finds the screwdriver's note, not the tweezers'.
- **Wonder fixing its own diff:** a diff that doesn't apply or compile goes back to Wonder once with the error and the build log, as in Reflexion's self-correction [5]. Measured: the LED's fix passes on the first try.
- **Every check before anything happens:** a diff must apply and build in a sandbox before it's staged, a flash waits for "go", and every pose of a pick and a put is solved before the first move. Moves are quintic, the minimum-jerk profile [15], and the IK is damped least squares [16]. Measured: 200 of 200 bench poses round-trip within 0.2 mm and 0.5°; after a move, each joint is re-aimed toward 0.2° of its target for up to three rounds.
- **A photo the read can trust:** each read measures its view's sharpness as the variance of the Laplacian [17] and how much glare blew out, and says what to change. Measured: good reads at 389–556 against a 150 warning.

## Results

Over the build, API costs came down by up to 75%, and the time spent waiting on models by about 90%. The changes behind them:

- **The whole conversation in every Spark call,** instead of chaining calls on `previous_response_id`: a follow-up call went from 12–14 s to about 2 s.
- **Bench facts gathered by code, kept locally and sent up front.** Before Spark's first call of a turn, code fetches `pins.h` and the board read in parallel, so Spark doesn't spend a round asking for them. The board read is made once at start-up and kept for the session, so later turns pay for neither the photo nor SAM.
- **Arm commands on a local model.** Plain rules and Llama 3.2 3B on this Mac, with its prompt cache, take an arm command in about 0.25 s, against about 4.4 s through Spark, and cost nothing.
- **Work started before it's needed:** the coder's sandbox is warmed at start-up (the fix compiles in about 8 s, not 14 s), the desk camera stays open so a photo doesn't wait for it to settle, and the arm powers on during the photo. Tool calls in one round run in parallel.

### In the video's takes (Nov 4)

The whole scene on the real bench, by voice: "Why isn't my LED on?", then "Go. Also, put my screwdriver in my cup, and add my screen to my schematic." Times are each tool's own, from 29 takes.

| Step | Median | Range |
|---|---|---|
| Spark finds the wire on the wrong GPIO, Wonder writes the fix and it compiles in the sandbox (`ask_coder`) | 9.2 s | 2.8–13.1 s |
| Build and flash (`build_and_flash`) | 18.1 s | 12.9–25.0 s |
| Spark Open on Baseten learns the screen, and KiCad's rules check passes on the drawn copy (`add_part`) | 11.4 s | 7.4–30.3 s |
| The arm stands the screwdriver in the cup (`move_thing`) | 42.0 s | 38.9–46.7 s |

A take costs about $0.06 in API calls.

## What runs live

Each piece runs with the real models, and the arm, the cameras and the demo board on the real bench.

| Piece | What it does |
|---|---|
| The LED fix | Reads the breadboard, finds the wire on the wrong GPIO against `pins.h`, has Wonder write and compile the fix, and flashes it on "go" |
| Router on Muse Spark | Routes each request in about 2 s a call, calling tools in parallel |
| Voice | Hands-free on the bench mic, every utterance transcribed by Muse Voice Transcribe; replies spoken by Kokoro-82M on the Mac in under a second |
| Arm fast path | Spoken arm commands decided on the Mac in about 0.25 s, by rules and Llama 3.2 3B agreeing |
| Picks and puts | "Put my screwdriver on my sticky note" in 31.5 s; "put my screwdriver in my cup" stands it in the mug on end |
| Board reader | Traces every flat jumper to its holes, builds the netlist and checks it against the firmware |
| Desk camera | Finds a named thing on the desk and places it in the arm's frame |
| Add a part | Spark Open on Baseten wires a new part to free GPIOs, saves it as a skill and draws it into the KiCad schematic, rules-checked |
| Spark Open | Answers questions no tool is needed for from the bench facts, in about 7 s |
| Memory | Picks up an earlier session and recalls where things go from notes |
| Kinematics | 200 of 200 bench poses round-trip within 0.2 mm and 0.5° |
| Budget guard | Every paid call checks tracked spend first |

## After the hackathon

- **12 V servos.** The SO-101 runs on 7.4 V STS3215s. From the URDF's link masses, reaching all the way out with 50 g held takes about 1.06 N·m at the shoulder lift, around or past the 50% torque cap on these servos, so a move out there can end in a sag cut instead of being refused up front. The 12 V STS3215 has about half again the torque. All six servos change together, with a 12 V supply and a new calibration: the current ones must never see 12 V (the shoulder lift's limit is 8.0 V).
- **Gravity-aware planning.** The same masses give each joint's gravity torque in any pose. Calibrated against the servos' own load readings, they would refuse a pose a joint can't hold before anything moves, and aim each joint past its target by its predicted sag, so a move needs fewer re-aim rounds.
- **Toolformer for every route.** After Meta's Toolformer, the local Llama learns from tool-call labels a model writes, kept only when the call runs, first for the fast path's arm commands, then for every route: whether a request needs a tool at all, which one, and when Spark Open can answer, replacing the word rules that pick a route today.
- **V-JEPA 2.** Meta's video world model, in two steps. First, its encoder on the desk camera's clip of a grip, telling a held thing from closed jaws, which the servos can't. Then its action-conditioned planner, trained on recorded runs of this arm, for moves the IK can't plan, such as turning a thing over before a put.
- **Skills it writes and keeps.** `add_part` saves what Spark Open works out about a part, but nothing reads it back yet: next it reuses a saved skill whose checks still pass. Then motion skills: a named routine of the arm's own moves, saved as data, every pose solved before it's saved and before each run.
- **Reinforcement learning, in simulation and then on the arm.** The URDF already carries every link's mass and inertia, so the arm can be simulated (MuJoCo, say) and policies trained there for what the written skills do least well: grasping odd shapes, standing a thing in the cup, recovering from a slip. Randomizing the desk, the lighting and the masses in simulation [18] is what lets a policy carry over to the real arm, where it would fine-tune on real runs with the reward read from the desk camera (SAM finding the thing where it should be), every action still under the driver's speed and torque caps and its cuts.
- **ROS 2.** Each skill as its own node, with actions for long moves (progress, and cancelling for the spoken stop), rosbag logs, tf2 for the camera and arm frames, and MoveIt's collision checks. The driver stays the bus's one owner.

## References

Meta's research first.

1. T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda and T. Scialom. *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
2. A. Kirillov et al. *Segment Anything.* ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643). And Meta's *SAM 3: Segment Anything with Concepts*, 2025, which SAM 3.1 builds on.
3. Llama Team, AI @ Meta. *The Llama 3 Herd of Models.* 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
4. M. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985). Future work: see [After the hackathon](#after-the-hackathon).
5. N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan and S. Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
6. J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang and M. S. Bernstein. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
7. L. Chen, M. Zaharia and J. Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* 2023. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
8. I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous and I. Stoica. *RouteLLM: Learning to Route LLMs with Preference Data.* 2024. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)
9. S. Kim, S. Moon, R. Tabrizi, N. Lee, M. W. Mahoney, K. Keutzer and A. Gholami. *An LLM Compiler for Parallel Function Calling.* ICML 2024. [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)
10. K. Spärck Jones. *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation 28(1), 1972.
11. S. Robertson and H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval 3(4), 2009.
12. K. Järvelin and J. Kekäläinen. *Cumulated gain-based evaluation of IR techniques.* ACM Transactions on Information Systems 20(4), 2002.
13. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
14. E. Olson. *AprilTag: A robust and flexible visual fiducial system.* ICRA 2011.
15. T. Flash and N. Hogan. *The coordination of arm movements: an experimentally confirmed mathematical model.* Journal of Neuroscience 5(7), 1985.
16. Y. Nakamura and H. Hanafusa. *Inverse kinematic solutions with singularity robustness for robot manipulator control.* Journal of Dynamic Systems, Measurement, and Control 108(3), 1986.
17. J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez and J. Fernández-Valdivia. *Diatom autofocusing in brightfield microscopy: a comparative study.* ICPR 2000.
18. J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba and P. Abbeel. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
