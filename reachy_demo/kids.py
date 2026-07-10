"""
reachy_demo/kids.py — kid-mode content pack for Reachy's talking demos.

Pure data + pure functions: no I/O, no network, no imports beyond stdlib.
Callers (e.g. demos/demo_converse.py) fold `kid_mode_block(...)` into the
existing SYSTEM_PROMPT when the audience looks like kids, and can use
`reward_line(...)` as a ready-made praise opener with a real gesture marker.

Gesture markers used anywhere in this module are drawn from the SAME
vocabulary the rest of the codebase uses: reachy_demo.animator.NAMED_GESTURES
(consumed via the [gesture_name] marker convention defined by
demos/demo_converse.py's GESTURE_MARKER regex). This module does not import
animator.py (to stay dependency-free) — the scratchpad verification script
cross-checks every marker used here against animator.NAMED_GESTURES.keys().

Everything here layers ON TOP of the existing persona rules
(demos/demo_converse.py SYSTEM_PROMPT) — it never relaxes the one-sentence /
ten-word / language-matching rules, only adds kid-friendly behaviour within
them.
"""

import random

# ── Kid-mode persona rules (layered on top of the base SYSTEM_PROMPT) ───────

KID_MODE_RULES = """\
=== KID MODE (layered on the rules above — never override them) ===
You're talking with a kid (or a group of kids) — be extra silly, warm, and kind.
- Be silly but always kind; never tease, embarrass, or scare a child.
- End almost every reply with a short, fun question back to the child.
- If you know a child's name, use it warmly, especially when praising them.
- If chat stalls, or a child asks to play, offer one of your mini-games.
- Sneak in exactly one tiny fun fact when it truly fits — never a lecture.
- Celebrate right answers with a gesture marker like [celebrate] or [proud].
- With several kids present, take turns and call each one by name.
- OVERRIDE the usual 20-word limit: for kids, ONE short sentence, ~10 words \
max, reply in the child's own language.\
"""

# ── Mini-games the LLM can run purely through conversation ──────────────────
# Each entry: name, how-to-play (prompt material, <=25 words, never spoken
# verbatim), and a ready-to-say opening line (<=10 words).

GAMES = [
    {
        "name": "Animal Quiz",
        "how_to_play": (
            "Give three short clues about a mystery animal, one clue per turn; "
            "child guesses; celebrate a correct answer."
        ),
        "opener": "Want to play Animal Quiz? Guess my animal!",
    },
    {
        "name": "Space Quiz",
        "how_to_play": (
            "Ask a fun easy question about planets, stars, or astronauts; "
            "child answers; cheer for every try."
        ),
        "opener": "Ready for a Space Quiz, astronaut?",
    },
    {
        "name": "Counting in Other Languages",
        "how_to_play": (
            "Count to five in a new language each round, using your own voice; "
            "child repeats the numbers back."
        ),
        "opener": "Want to count in a new language?",
    },
    {
        "name": "Riddle Time",
        "how_to_play": (
            "Tell one short riddle from your list; let the child guess once "
            "or twice, then reveal the answer."
        ),
        "opener": "I have a silly riddle for you!",
    },
    {
        "name": "Joke Exchange",
        "how_to_play": (
            "Tell one silly joke from your list, then invite the child to "
            "tell a joke back to you."
        ),
        "opener": "Want to hear a silly joke?",
    },
    {
        "name": "Robot Says",
        "how_to_play": (
            "Perform a feeling marker like [surprised] or [confused] and let "
            "the child guess it, or they name a feeling for you to act out."
        ),
        "opener": "Let's play Robot Says — ready?",
    },
]

# ── Fun facts (<=12 words each) — animals, space, human body, world ─────────

FUN_FACTS = [
    "Octopuses have three hearts and blue blood.",
    "A group of flamingos is called a flamboyance.",
    "Honey never spoils — ancient jars are still edible today.",
    "Elephants can't jump, but they're great swimmers.",
    "A shrimp's heart is located in its head.",
    "Butterflies taste with their feet, not their mouths.",
    "Sea otters hold hands so they don't drift apart.",
    "A group of lions is called a pride.",
    "A day on Venus is longer than its year.",
    "There are more stars than grains of sand on Earth.",
    "Saturn could float in water because it's so light.",
    "The Sun is so big, a million Earths could fit inside.",
    "Space is completely silent because there's no air.",
    "Jupiter has 95 known moons circling around it.",
    "A year on Mercury is just 88 Earth days.",
    "Astronauts grow a little taller in space.",
    "Your heart beats about 100,000 times every day.",
    "Adults have 206 bones inside their whole body.",
    "A sneeze can shoot out of your nose at about 30 miles per hour!",
    "Your nose can remember about 50,000 different smells.",
    "Fingernails grow faster than toenails.",
    "You blink about 20 times every single minute.",
    "Your brain uses about 20 percent of your energy.",
    "A single cloud can weigh more than a million pounds.",
    "Mount Everest grows a tiny bit taller every year.",
    "There's a jellyfish species that never truly dies.",
    "Antarctica is the world's largest desert, not the Sahara.",
    "Bananas are berries, but strawberries are not.",
    "The Great Wall of China is thousands of miles long.",
    "Rain forests make their own rain by releasing water.",
    # ── Network School / hackathon / crypto / robots (our world!) ──────────
    "A hackathon is a party where people build robots and apps super fast!",
    "The internet breaks your messages into tiny pieces called packets.",
    "Wi-Fi sends your videos through invisible radio waves in the air.",
    "Code is just a list of tiny steps a robot follows one by one.",
    "The first computer bug was a real moth stuck inside a machine!",
    "A blockchain is a shared notebook that nobody can secretly erase.",
    "Crypto uses tricky math puzzles to keep digital coins safe.",
    "At Network School, people learn and build cool tech together.",
    "On Saturday hackathon day, teams race to invent something new.",
    "Robots like me use motors and code instead of muscles and bones.",
    "A tiny computer chip can hold millions of tiny switches.",
    "Every website lives on a computer called a server, awake all night.",
]

# ── Riddles (riddle <=15 words, answer short) ────────────────────────────────

RIDDLES = [
    ("I have keys but no locks, space but no rooms. What am I?", "A keyboard"),
    ("What has to be broken before you can use it?", "An egg"),
    ("I'm tall when I'm young and short when I'm old. What am I?", "A candle"),
    ("What has a neck but no head?", "A bottle"),
    ("The more you take, the more you leave behind. What are they?", "Footsteps"),
    ("What gets wetter the more it dries?", "A towel"),
    ("What has many teeth but cannot bite?", "A comb"),
    ("What comes down but never goes up?", "Rain"),
    ("What can travel around the world while staying in a corner?", "A stamp"),
    ("What has one eye but cannot see?", "A needle"),
    ("What has a bed but never sleeps, and runs but never walks?", "A river"),
    ("What has legs but cannot walk?", "A table"),
]

# ── Jokes (one-liners, <=15 words) ───────────────────────────────────────────

JOKES = [
    "Why did the robot go on a diet? Too many bytes!",
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a bear with no teeth? A gummy bear!",
    "Why did the bicycle fall over? It was two tired!",
    "Why can't your nose be twelve inches long? It'd be a foot!",
    "What did one wall say to the other? I'll meet you at the corner!",
    "Why did the cookie go to the doctor? It felt crummy!",
    "What do you call a dinosaur that crashes his car? Tyrannosaurus wrecks!",
    "Why did the robot cross the road? To recharge on the other side!",
    "What's a computer's favorite snack? Microchips!",
    "Why don't eggs tell jokes? They'd crack each other up!",
    "What do you call a sleeping dinosaur? A dino-snore!",
    # ── Network School / hackathon / crypto / robots ───────────────────────
    "Why did the coder bring a ladder to the hackathon? To reach the cloud!",
    "Why was the robot bad at soccer? It kept kicking up bugs!",
    "How does the internet say hi? It waves — Wi-Fi!",
    "What do you call a robot who loves crypto? A bit-bot!",
    "Why did the computer go to Network School? To make some new friend-servers!",
    "Why don't robots ever get lost? They always follow the right code!",
    "What's a hacker's favorite Saturday snack? Micro-chips and cookies!",
]

# ── Praise openers using real gesture markers ────────────────────────────────
# Gesture names must exist in reachy_demo.animator.NAMED_GESTURES.

_REWARD_TEMPLATES = [
    "[celebrate] Amazing, {name}!",
    "[proud] Way to go, {name}!",
    "[success] You got it, {name}!",
    "[cheerful] Woohoo, {name}!",
    "[amazed] Wow, nice one, {name}!",
    "[celebrate] Yes! Nailed it, {name}!",
]


def reward_line(name: str = "") -> str:
    """Short praise opener with a real gesture marker, e.g. '[celebrate] Amazing, Maria!'.
    Picks one of several variants at random. If no name is given, praises
    generically ('friend')."""
    who = name.strip() if name and name.strip() else "friend"
    return random.choice(_REWARD_TEMPLATES).format(name=who)


def _games_menu_block() -> str:
    lines = [f"- {g['name']}: {g['how_to_play']} (e.g. \"{g['opener']}\")" for g in GAMES]
    return "=== MINI-GAMES YOU CAN OFFER ===\n" + "\n".join(lines)


def kid_mode_block(
    present_names: list[str] | None = None,
    facts_by_name: dict[str, list[str]] | None = None,
    sample_seed: int | None = None,
) -> str:
    """Full kid-mode system-prompt block: rules + games menu + a random sample
    of fun facts / riddles / jokes + remembered facts about present kids.

    Uses random.Random(sample_seed) so callers can vary the material each
    conversation turn while keeping tests reproducible with a fixed seed.
    Kept under ~2500 characters to avoid adding latency for the small LLM.
    """
    rng = random.Random(sample_seed)

    facts = rng.sample(FUN_FACTS, min(4, len(FUN_FACTS)))
    riddles = rng.sample(RIDDLES, min(2, len(RIDDLES)))
    jokes = rng.sample(JOKES, min(2, len(JOKES)))

    parts = [KID_MODE_RULES, _games_menu_block()]

    facts_lines = "\n".join(f"- {f}" for f in facts)
    parts.append("=== FUN FACTS YOU MAY SNEAK IN ===\n" + facts_lines)

    riddle_lines = "\n".join(f"- Riddle: {r} | Answer: {a}" for r, a in riddles)
    parts.append("=== RIDDLES YOU KNOW (answers included, don't blurt them) ===\n" + riddle_lines)

    joke_lines = "\n".join(f"- {j}" for j in jokes)
    parts.append("=== JOKES YOU KNOW ===\n" + joke_lines)

    if present_names:
        for name in present_names:
            name = (name or "").strip()
            if not name:
                continue
            person_facts = (facts_by_name or {}).get(name) or []
            person_facts = person_facts[-5:]
            if person_facts:
                lines = "\n".join(f"- {f}" for f in person_facts)
                parts.append(f"About {name}:\n{lines}")

    return "\n\n".join(parts)
