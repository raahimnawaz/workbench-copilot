# HUD orb: what it borrows

`orb-light.html` is the HUD's orb page, and `saved/siri-line.js` is the Siri line it loads. The hackathon rule in `CLAUDE.md` is that the demo may only show work built during the event, and that the README's built-vs-reused table lists any reused code with its license. This file says what the page borrows, and whether that's code or only a look.

## Code adapted: each has a line in the README's reused table

**thinking-orbs**, by Jakub Antalik. MIT License, Copyright (c) 2026 Jakub Antalik. <https://github.com/Jakubantalik/thinking-orbs>

- What's taken: the dot lattice, rings of dots by latitude on a sphere, from its `src/engine/lattice.ts`.
- What's changed: the sphere turns on the Siri line as its axis, and the dots move with our own ocean waves instead of its motions.
- The full MIT notice is at the end of `orb-light.html`.

**WebGL-Fluid-Simulation**, by Pavel Dobryakov. MIT License, Copyright (c) 2017 Pavel Dobryakov. <https://github.com/PavelDoGreat/WebGL-Fluid-Simulation>

- What's taken, from `script.js`:
  - the bloom: the prefilter, the blur chain and the final pass, with `applyBloom`
  - the sun rays: the rays shader with `applySunrays`, and the separable blur
  - the vertex shaders, the quad and `blit`, and the framebuffer set-up
- What's changed:
  - 8-bit textures instead of half floats
  - the rays' mask turned round, so the bright dots make the rays, streaming outward from the orb's middle
  - the rays in the dots' own colors (the mask and the sum in color instead of one channel), made a little richer, like light through stained glass
  - the rays without each point's own light
  - our own final pass, adding the bloom and the rays to the page (the dots themselves are drawn on the orb)
- Where it's used: the dots' light. The full MIT notice is at the end of `orb-light.html`.

## Looks recreated from scratch: credit as inspiration, no code copied

- **libraries.dev Voice Glow** by Jakub Antalik, the `voice-glow` package (some features are Pro). <https://libraries.dev/voice>
  - The glow that rises with the voice.
  - Here it's written fresh to light the dots and the line's waves, in the orb's own colors (the primary blue, the model's color and Muse blue) instead of their palette. We didn't check its license, because nothing was copied.
- **libraries.dev Border Beam**. <https://libraries.dev/beam>
  - The beam riding the border of the text and file box.
  - Here it's a CSS conic gradient in the orb's colors.
- **libraries.dev Metal** (Liquid Metal, the `metal-fx` package, built on Paper Shaders). <https://libraries.dev/metal>
  - The live metal is our own WebGL shader, written from the look of their live demo, with no code from metal-fx or Paper Shaders:
    - flowing bands, with each color channel reading them a little apart so opal edges show, plus fine opal streaks and a thin thread of the model's color
    - the words: the model's name in white, edged in black metal that glints in the model's colors, and "on Baseten" in bright green metal with a green glow; the same shader draws the reply, white with a touch of light orange and a glowing blue outline
    - the send button: black metal with a silver outline and a silver arrow, whose light takes the orb's colors while it's pressed
  - Without WebGL, the words fall back to a CSS gradient clipped to the text.
- **VoiceOrbs** by amunozdev, MIT. <https://github.com/amunozdev/voiceorbs>
  - Its Siri Wave Line gave the idea for the Siri line in `saved/siri-line.js`.
  - Everything here is written from scratch.
- **Apple's Liquid Glass** (WWDC 2025 and the Human Interface Guidelines' Materials page).
  - The glass panes behind the model's name and the reply, and the glass of the box and its buttons.
- **Meta Connect 2026's key art**.
  - The grain over the dark background.
- **OKLab and OKLCH** by Björn Ottosson. <https://bottosson.github.io/posts/oklab/>
  - The published conversion constants are used for the contrast color's hue math.
- **Textbook ocean-wave equations**, used as formulas, with all the code written fresh:
  - deep water's dispersion, ω² = g k
  - Gerstner's trochoidal waves (1802)
- **A reference image shared in the design chat**: a sphere of dots colored in sections.
  - It inspired the dots' colors in sections.
  - Nothing is taken from it.

## Looked at, nothing taken

- [aaaa-zhen/glsl-playground](https://github.com/aaaa-zhen/glsl-playground)
- [deepika-builds/liquid-glass](https://github.com/deepika-builds/liquid-glass)
- [lucasromerodb/liquid-glass-effect-macos](https://github.com/lucasromerodb/liquid-glass-effect-macos) (no license)
- the [ElevenLabs UI](https://elevenlabs.io/blog/elevenlabs-ui) Orb
- for light through stained glass: [Erkaman's glsl-godrays](https://github.com/Erkaman/glsl-godrays) and the pmndrs godrays for three.js; the colored rays here are the ported sun rays above, changed to carry color

## Ours

- Everything else is written for this project:
  - the saved Siri line (`saved/siri-line.js`)
  - the spear the sphere turns on, the dots parting round it, and the line dimmed inside the sphere
  - the sea states, which stage gets which, and how they blend
  - the dots as clear light in the Siri line's own look, joined by faint white lines that stop at their edges
  - the colors in sections by how squarely each dot faces you, rippling with the waves, with the contrast color (the primary's and the model's hues averaged and turned half round) warming the rim
  - the light behind and along the line's waves, with a little gold at its heart, and the line kept the brightest thing on the page
  - the glass pane behind the model's name, and the reply's glass bubble that grows as sentences come in, with a faint gradient outline that reads more like a shadow than a line
  - the live liquid glass: the sweeping sheen and the drifting lamp
  - the sky's neural map with its aurora-like links, and the stars
  - the beam's pacing, and the send button's reflection trailing its head
  - the call bar: each model enters as it starts, with a dot like a tiny orb in that model's colors
  - the page around the orb
- No npm packages and no font files: it's plain HTML with canvas and WebGL, in the system font.
