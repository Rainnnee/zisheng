# Demo files

The root `assets/` files preserve the original three-character reference run. The expanded 12-character, 48×48 qualitative run is in [`assets/expanded-12char-48px/`](assets/expanded-12char-48px/), alongside its checkpoint and run configuration.

Open [`assets/index.html`](assets/index.html) to inspect the sample gallery, selected reverse-process states, and training trace. The outputs are model samples from a synthetic printed-font pilot; they have not been scored by an OCR system or evaluated on held-out fonts.

For an interactive local view that generates fresh samples from the expanded checkpoint, start the website from the project root:

```bash
python -m zisheng web
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) if the browser does not open automatically. Choose among 12 characters, set a random seed and sample count, and watch actual model states arrive during generation. The sampler sends a frame every 10 steps; each sample is a separate image, and selecting one focuses the timeline on that sample. After generation, use the slider or play the states at 30 frames per second. Leave the terminal open while the page is in use; press `Ctrl+C` to stop the server. This uses the saved checkpoint and does not train the model again.

To reproduce training and create a fresh report, run the command in the root README. To sample from the saved checkpoint without retraining:

```bash
python -m zisheng sample --checkpoint demo/assets/expanded-12char-48px/zisheng-pilot.pt --output results/runs/new-samples --open
```
