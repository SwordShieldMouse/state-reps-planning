# state-reps-planning
This rep is a playground for me to explore ideas about knowledge representation in the context of developing an AI. Most of these ideas are obtained/inspired by Rich Sutton's reinforcement learning class and text.

Incomplete list of ideas explored:
- General Value Functions
  - Using GVFs as features
  - Learning a GVF that estimates time till episode end in cart pole
- Generating features that are tested and thrown away according to how useful they are in achieving reward
- Adaptive step-sizes per feature that are updated through some gradient descent rule
  - Go one step forward and run descent on estimated TD error for next step (does not really work)
