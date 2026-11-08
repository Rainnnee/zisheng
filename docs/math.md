# Diffusion mathematics used in the pilot

## Forward corruption

Let $x_0$ be a clean, 48×48 grayscale glyph in the current expanded pilot. The renderer scales paper pixels to $+1$ and ink pixels to $-1$. At step $t$, the forward Markov transition is

$$q(x_t\mid x_{t-1})=\mathcal{N}\!\left(\sqrt{\alpha_t}x_{t-1},\,\beta_t I\right),\qquad \alpha_t=1-\beta_t.$$

Multiplying the one-step transitions gives $\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$ and the closed-form marginal

$$q(x_t\mid x_0)=\mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}x_0,\,(1-\bar{\alpha}_t)I\right).$$

It can be sampled directly as

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).$$

The training code uses this marginal, so it can choose a random timestep without simulating the preceding steps. The pilot uses 1,000 steps and a linear beta schedule from $10^{-4}$ to $0.02$. At the final step $\bar{\alpha}_T$ is close to zero, making the terminal distribution close to standard Gaussian noise.

## Conditional noise prediction

The denoiser receives the noisy image $x_t$, the timestep $t$, and a character class $c$. It predicts the particular noise tensor added to the clean image:

$$\epsilon_\theta(x_t,t,c).$$

The pilot minimizes the standard squared-error objective

$$\mathcal{L}=\mathbb{E}_{x_0,\,c,\,t,\,\epsilon}\left[\left\|\epsilon-\epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\,t,\,c\right)\right\|_2^2\right].$$

This is a training objective. A lower value does not by itself establish that generated images are recognizable or generalize to new fonts.

## Reverse sampling

At each step, the predicted noise defines the DDPM posterior mean

$$\mu_\theta(x_t,t,c)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,c)\right).$$

For $t>1$, the sampler adds Gaussian noise with posterior variance

$$\tilde{\beta}_t=\beta_t\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}.$$

At the last reverse step, it returns the posterior mean without adding noise. Sampling begins from $x_T\sim\mathcal{N}(0,I)$ and runs the learned reverse transitions down to $x_0$.

## Why different seeds change the apparent stroke size

The class condition identifies the character, but it does not specify font size or line weight. The training renderer varies font size from 36 to 42 pixels, adds up to one native pixel of stroke weight, shifts glyphs by up to two pixels, rotates them by up to five degrees, and uses both Sans and Serif fonts. These are deliberate variations in the distribution the model learns.

At sampling time, a seed changes the starting Gaussian field $x_T$ and the Gaussian noise used in intermediate reverse steps. Different seeds therefore produce different samples from the learned class distribution. The model is not trying to recover one canonical outline, and the present implementation has no scale or weight control. A future controlled-size version could normalize glyph bounds or add scale and weight as explicit conditions.

The U-Net operates on a grayscale pixel array. Pillow renders the training examples from licensed fonts, while the browser enlarges each generated pixel array for inspection; the model does not predict pen paths or crop samples out of a contact sheet.

## Where the equations appear in the code

- `src/zisheng/diffusion.py`: precomputes the schedule, constructs $x_t$, computes the noise-prediction loss, and applies the reverse mean and variance.
- `src/zisheng/model.py`: embeds timestep and character ID, then predicts the noise image with a compact U-Net.
- `src/zisheng/train.py`: samples timesteps and noise, updates the denoiser, and maintains an exponential moving average for sampling.
- `src/zisheng/sample.py`: loads the saved weights and records selected states in the reverse chain.

The formulation follows Ho, Jain, and Abbeel, [“Denoising Diffusion Probabilistic Models” (2020), Section 2](https://arxiv.org/abs/2006.11239). The first model demo implements the forward and reverse processes; it does not yet analyze recognition across corruption levels.
