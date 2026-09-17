# Paper landing page

This folder is ready to use as a GitHub Pages site for:

**Observing and Evading Quantum Back-Action on a Kilogram-Scale Oscillator**

## Publish it

1. Upload `index.html` to the root of your public GitHub repository.
2. In the repository, go to **Settings → Pages**.
3. Under **Build and deployment**, choose:
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/(root)`
4. Save.

Your stable URL will look like:

`https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/`

That is the URL you can give your advisor now.

## Add the figure

Put your screenshot or main paper figure in the repository root and name it:

`hero-figure.png`

If the file is absent, the figure section hides automatically.

## When the arXiv page becomes public

Open `index.html` and find the disabled `arXiv link pending` button. Immediately above it is a commented replacement button.

Replace it with:

```html
<a class="btn btn-primary" href="https://arxiv.org/abs/XXXX.XXXXX">View on arXiv</a>
```

Also update the **Preprint** resource card near the bottom.

The GitHub Pages URL itself does not change.

## Optional links

There are placeholders for:
- analysis / figure code
- public data / DOI

Replace each `href="#"` with the public URL when ready.
