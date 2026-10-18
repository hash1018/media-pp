# What measuring speech turned up

The first runs of the speech benchmark measured a transcriber that lost
words, and most of what they found was fixed before the numbers in
[windows-rtx3050.md](windows-rtx3050.md) were taken. This is what each run
showed, what was done about it, and what was tried and dropped — measured
with `large-v3-turbo` on CUDA unless said, on the FLEURS clips
[method.md](method.md) describes.

## Rounds cut by time lost words at every seam

The transcriber heard four seconds at a time, each round with the one
before it again as context, and kept the words the model timed inside its
own window. The model times the same word hundreds of milliseconds apart
in two rounds, so a word at the seam fell in neither window or in both:
"재입국 충[격은 신혼 단계가 없]기", "빠르게 / 빠르게".

| Korean, long clip | Heard at once | Streamed | Streamed, words aligned |
|---|---:|---:|---:|
| CER before | 3.3% | 25.0% | 9.6% |

A word is now reported once two rounds in a row agree on the words up to
it, by their text (LocalAgreement, after Macháček et al.'s
whisper_streaming), and what is reported stays in the audio heard until it
is cut away and given to the model as its prompt (8724cdb). Streamed, the
clip then missed 7.5% of its characters, at 6x real time where it had been
7x: the model now hears up to 25 seconds every round rather than eight,
though Whisper's encoder costs the same for either.

## The model repeated itself, and every round agreed

The first full run put `medium` on Vulkan at 40% CER on the Korean clip,
where heard at once it missed 7.5%. One line the model made up — "이
인공위성을 찾을 수 있는 기술을 지키는 것입니다" — was committed, went into
the prompt, and was then said again every round with growing confidence:
fifteen times, over five sentences that were never heard. Its average
token log-probability climbed from −0.5 to −0.03 as it repeated, so a
confidence threshold, which is how Whisper itself refuses a bad window,
would have kept it.

A run of four words that the last sixty committed words already hold is
now passed over, and the prompt dropped for the next round, which hears the
audio alone (ef824cd). Refusing the whole round instead lost the words
around a short echo and, once, the end of the stream with them.

Two other changes were tried against the same transcripts and dropped:

| Korean CER, long clip, 8724cdb | base | small |
|---|---:|---:|
| As it was | 32.5% | 27.8% |
| Temperature fallback off | 160.6% | 51.2% |
| No prompt | 19.5% | 32.3% |

Without the fallback, which re-decodes a window that compresses too well
or that the model is unsure of, a small model loops outright and never
stops. Without the prompt it went both ways, and on a later build with the
cuts below, it took `medium` from 10.8% to 18.3% and turbo from 4.4% to
10.2%: a committed word the cut left audible is said again, since nothing
tells the model it was said.

## Cuts fell inside words

A small model's word times are a second out. The audio was cut halfway
between the two committed words the model said part, which put the cut
inside one of them often enough, and whisper, handed a round that begins
with the end of a word, passed over the sentence it began — two rounds
running, so that the passing over was agreed on. The cut now goes to the
quietest 40 ms within 600 ms of that guess, never into the second committed
word it keeps, and the last eight words cut away are still lined up
against, in case one is heard again (ef824cd).

## Two rounds agreed that a sentence was silence

The next full run put `large-v3` at 13% where it had been 5%: the last two
sentences were gone. Whisper had passed over a sentence begun near the end
of what it heard, two rounds running, and two rounds hearing nothing new
is what settled a stretch as silence — nine seconds of speech. The last
round then heard only the echo of a committed word and stopped, and the
twenty words the round before had heard went with it.

Silence is now believed only where the audio is quiet too: no 100 ms of it
within 30 dB of the loudest heard. Where the model heard nothing in more
than two seconds of sound after its last word, that tail is heard again on
its own, without the prompt; and the last round commits whichever of its
hearing and the one before holds more (9ac0142).

## A clip of 26 sentences is too few to judge a change by

Whisper passes over a whole sentence now and then, heard at once or
streamed, and which one it passes over moves with any change: a different
cut, one word of prompt more. On a clip of 26 sentences one of them is
eight points of WER. Between ef824cd and 9ac0142 the benchmark's English
clip said `base` had gone from 14% to 23%; ten minutes a language of
sentences the benchmark clips do not use, made the same way, said the
opposite:

| 10 min held out | small, Korean CER | small, English WER | medium, Korean | medium, English | turbo, Korean | turbo, English |
|---|---:|---:|---:|---:|---:|---:|
| ef824cd | 22.7% | 20.4% | 19.4% | 20.0% | 5.6% | 11.4% |
| 9ac0142 | 15.5% | 13.1% | 12.2% | 12.3% | 5.5% | 8.2% |

A change to the transcriber is judged on clips like these, not on the
benchmark's own.

## Half of FLEURS's English recordings are all but silent

The first English numbers had the whole clip heard at once missing 73% of
its words. Four of the seven sentences in the short clip were recorded at
−60 to −67 dBFS, 40 dB below the rest, and whisper passed each over whole,
at once and streamed alike. Most FLEURS sentences are read by two
speakers; `make_clips.py` takes the louder recording, and leaves out a
sentence whose louder one is still below −55 dBFS — 21 of the first 47
English sentences, 3 of the first 26 Korean.

## A mono WAV made no sound at all

The clips are WAV files with no channel mask, which FFmpeg decodes to a
channel layout that says how many channels and not which. libswresample
takes such a layout as the default for that many when it builds a context,
then refused every frame still saying "unspecified" as an input that had
changed: `AudioResampler` produced nothing, and the transcriber heard
nothing (4005792). Such a frame is now given the default layout first;
`SwAudioEncoder` and `AudioMixer`'s inputs, which resample through the same
code, were fixed with it.
