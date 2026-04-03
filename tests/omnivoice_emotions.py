#!/usr/bin/env python
"""
Simple OmniVoice inline emotion test for RealtimeTTS.

- Loads emotion reference WAVs from: emotional_wavs/
- Registers each one as an inline tag
- Speaks one single tagged block of text

Run from repo root or tests folder:
    python tests/omnivoice_simple_emotions.py
"""

from pathlib import Path

from RealtimeTTS import TextToAudioStream, OmniVoiceEngine, OmniVoiceVoice


BASE_DIR = Path(__file__).resolve().parent / "emotional_wavs"
LANGUAGE = "en"

# Exact transcripts of the reference WAVs
REF_TEXTS = {
    "neutral": (
        "My coffee turned out perfect this morning, and I've got a good feeling about today. "
        "The sun is finally out and I've got nowhere to be, so I'm just going to enjoy the walk."
    ),
    "calm": (
        "Take a slow breath and look at me for a second. Nothing is falling apart "
        "yet, and we can handle this one step at a time without panicking."
    ),
    "cheerful": (
        "There you are! I was hoping today would start with something good, and "
        "seeing you now somehow makes the whole morning feel lighter already."
    ),
    "excited": (
        "Wait, seriously? We actually pulled it off, and after everything that "
        "almost went wrong, it finally worked exactly the way we hoped it would!"
    ),
    "sadness": (
        "I kept thinking you might still come back somehow, even long after I had "
        "every reason to understand that some things do not return once they are gone."
    ),
    "anger": (
        "Don't stand there trying to twist this around on me again. Just look me "
        "in the eye for once and tell me the truth without hiding behind excuses."
    ),
    "nervous": (
        "Did you hear that just now? I swear there is something moving out there, "
        "and I really don't think we should stay here any longer."
    ),
    "anxious": (
        "I know it will probably be fine in the end, and maybe I'm overthinking "
        "it again, but I still can't shake this awful feeling in my chest."
    ),
    "confident": (
        "Listen to me carefully for a moment. This is the right move, and I know "
        "exactly why it works, even if it doesn't make sense to everyone yet."
    ),
    "curious": (
        "That's really interesting, actually. What do you think would happen if we "
        "approached it from the other side and tried something slightly different this time?"
    ),
    "surprised": (
        "Wait, you knew all of this the whole time and never said a word to anyone? "
        "How was I supposed to see that coming?"
    ),
    "determined": (
        "We have already come too far, sacrificed too much, and worked too hard to "
        "stop now just because the last part is difficult."
    ),
    "compassionate": (
        "I know this hurts more than you want to admit, and you really do not have "
        "to carry all of it alone while pretending you're fine."
    ),
    "gently": (
        "Hey... it's alright now. You don't have to explain everything tonight, "
        "and you don't have to be strong for me every second."
    ),
    "playfully": (
        "Oh, so that's your grand master plan? I have to admit, it's a little "
        "reckless, a little ridiculous, and somehow exactly your style."
    ),
    "romantic": (
        "When you look at me like that, the whole world seems to go quiet for a "
        "moment, like nothing else matters except being here with you."
    ),
    "sarcastic": (
        "Oh, fantastic. Because clearly this day was going far too smoothly, and "
        "what it really needed was one more completely avoidable disaster."
    ),
    "urgent": (
        "Move, now. We are almost out of time, and if we wait even a few more "
        "seconds, this is going to get much worse."
    ),
    "whisper": (
        "Keep your voice down and don't turn around too quickly. If someone is "
        "listening from the other side of that door, I don't want them hearing us."
    ),
}

# Demo lines to synthesize with inline emotion tags
DEMO_TEXTS = {
    "neutral": (
        "Hello. I am online and ready to help you with whatever strange chaos you brought me today."
    ),
    "calm": (
        "Take a slow breath first. We can untangle this neatly, one quiet step at a time."
    ),
    "cheerful": (
        "Well, good morning to us. Somehow your arrival made this whole session feel much nicer already."
    ),
    "excited": (
        "Wait, no way, we actually nailed it. After all that chaos, it finally worked exactly like we wanted."
    ),
    "sadness": (
        "Some bugs do not just break code. They break the tiny bit of hope we were still holding onto."
    ),
    "anger": (
        "Do not blame the server again. Look at this mess and tell me with a straight face that this was a good idea."
    ),
    "nervous": (
        "Did you see that flicker in the logs? I really do not like it when a system starts acting alive at night."
    ),
    "anxious": (
        "It will probably be fine, right? And yet i still have that horrible feeling that something is about to go wrong."
    ),
    "confident": (
        "Listen carefully. This is the right fix, and in a minute you will see exactly why i was right."
    ),
    "curious": (
        "Now that is interesting. What happens if we flip the whole approach and poke the problem from the other side?"
    ),
    "surprised": (
        "Wait, you had that hidden the whole time? That explains far too much and also raises several new concerns."
    ),
    "determined": (
        "We are not stopping here. We came too far, fought too hard, and i refuse to lose to one stubborn final problem."
    ),
    "compassionate": (
        "Hey, i know this is exhausting. You do not have to carry the whole mess alone and pretend it does not hurt."
    ),
    "gently": (
        "It's alright. You do not need to explain everything tonight. Just stay here for a second and breathe."
    ),
    "playfully": (
        "Oh, that is your master plan? Wild, reckless, slightly ridiculous... and honestly very entertaining."
    ),
    "romantic": (
        "When you stare at the screen with me like that, even a clean stack trace starts feeling weirdly intimate."
    ),
    "sarcastic": (
        "Oh, fantastic. Because obviously this perfectly normal day needed one more absurd and completely avoidable disaster."
    ),
    "urgent": (
        "Move now. Save everything, close the window, and do not give this situation one more second to get worse."
    ),
    "whisper": (
        "Keep your voice down. If the bug is hiding in this file, i do not want to scare it back into the walls."
    ),
}


def make_voice(name: str) -> OmniVoiceVoice:
    wav_path = BASE_DIR / f"{name}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing file: {wav_path}")

    return OmniVoiceVoice(
        name=name,
        ref_audio=str(wav_path),
        ref_text=REF_TEXTS[name],
        language=LANGUAGE,
    )


def build_demo_block() -> str:
    order = [
        "neutral",
        "calm",
        "cheerful",
        "excited",
        "sadness",
        "anger",
        "nervous",
        "anxious",
        "confident",
        "curious",
        "surprised",
        "determined",
        "compassionate",
        "gently",
        "playfully",
        "romantic",
        "sarcastic",
        "urgent",
        "whisper",
    ]
    return " [pause=0.15] ".join(f"[{name}] {DEMO_TEXTS[name]}" for name in order)


if __name__ == "__main__":
    base_voice = make_voice("neutral")

    engine = OmniVoiceEngine(
        voice=base_voice,
        debug=True,
    )

    stream = TextToAudioStream(engine)

    stream.add_pause("pause", duration=0.5)

    for name in REF_TEXTS:
        stream.add_voice(name, make_voice(name))

    text = build_demo_block()

    print("\n--- TAGGED DEMO TEXT ---\n")
    print(text)
    print("\n--- START ---\n")

    stream.feed(text).play(
        log_synthesized_text=True,
        fast_sentence_fragment=False,
        force_first_fragment_after_words=9999,
        minimum_sentence_length=20,
        minimum_first_fragment_length=20,
        comma_silence_duration=0.12,
        sentence_silence_duration=0.25,
        default_silence_duration=0.15,
    )

    engine.shutdown()
    print("\nDone.")
