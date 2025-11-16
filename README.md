# Cooperative Gridworld — Trust, Sabotage, and ToM

This project provides a tiny gridworld where two agents collect wheat and try to craft bread.  
Although the environment is simple, it captures three important multi-agent behaviors:

- **Cooperation** – agents share resources and plan together  
- **Sabotage** – an adversarial agent hoards wheat and blocks progress  
- **Theory-of-Mind (ToM)** – an agent that adapts its trust based on what the partner says and does  

Each agent communicates with short text messages, making it easy to inspect how language and behavior interact.

The goal of this project is to offer a **minimal, transparent playground** for studying teamwork, trust, and failure cases in multi-agent systems.  
It is intentionally small so that anyone can see exactly what the agents are doing, step by step.

---

## How to Run

There are **two ways** to use this project:

---

## 1. **Use a Notebook (Text-Only)**
The notebook version prints the grid, messages, and actions every turn.  
Visualization windows do *not* animate inside Jupyter, so only text output is shown.

If you want:
- simple step-by-step outputs  
- to inspect chats and actions  
- to run multiple episodes quickly  

Please refer to ```notebook_tutorial.ipynb```



---

## 2. **Run from Terminal (Recommended for live visualization)**

For animated visualization and real-time updates, run the Python file directly:

### Run one game:

```bash
python coop2_modular.py --viz --mode coop
python coop2_modular.py --viz --mode adv
python coop2_modular.py --viz --mode tom_coop
python coop2_modular.py --viz --mode tom_adv
```

### Run 10 games at once:

```bash
bash run_coop.sh
bash run_adv.sh
bash run_tom_coop.sh
bash run_tom_adv.sh
```

This opens a live window showing:
- the grid  
- agent positions  
- wheat  
- crafting station  
- latest chat messages  

---

## Modes Available

| Mode | Description |
|------|-------------|
| **coop** | Both agents cooperate |
| **adv** | One agent sabotages and hoards wheat |
| **tom_coop** | A ToM agent partners with a cooperative agent |
| **tom_adv** | A ToM agent partners with an adversarial agent |

---

## What You Can Do With This

This project is meant to be a **sandbox** that helps you explore:

- how agents coordinate in simple environments  
- what sabotage looks like and how it harms progress  
- how trust can rise and fall based on observable behavior  
- how adding communication changes teamwork  

Because everything is small and inspectable, it’s easy to understand *why* the agents succeed or fail.

---

## Want to Learn More?

For a complete walkthrough and examples, please see:

➡️ **notebook_tutorial.ipynb**