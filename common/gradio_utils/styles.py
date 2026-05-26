from __future__ import annotations

APP_CSS = """
:root {
  color-scheme: light;
  --lance-accent: #fb923c;
  --lance-accent-hover: #f97316;
  --lance-surface: #ffffff;
  --lance-surface-muted: #f8fafc;
  --lance-border: rgba(148, 163, 184, .36);
  --lance-text: #111827;
  --lance-text-muted: #475569;
  --lance-shadow: 0 8px 24px rgba(15, 23, 42, .08);
  --body-background-fill: var(--lance-surface);
  --background-fill-primary: var(--lance-surface);
  --block-background-fill: var(--lance-surface);
  --input-background-fill: var(--lance-surface);
  --button-primary-background-fill: var(--lance-accent);
  --button-primary-background-fill-hover: var(--lance-accent-hover);
  --button-primary-text-color: #0f172a;
}
body, .gradio-container, .contain { background: var(--lance-surface) !important; color: var(--lance-text) !important; }
.gradio-container, .contain { max-width: 1180px !important; margin: 0 auto !important; }
.lance-hero { text-align: center; padding: 8px 12px 4px; }
.lance-logo { width: min(150px, 34vw); height: auto; display: block; margin: 0 auto 4px; }
.lance-title { margin: 0 auto 5px; font-size: clamp(22px, 2.4vw, 32px); line-height: 1.08; font-weight: 800; color: var(--lance-text) !important; }

/* Keep key titles/labels above the white content panels so they are never covered. */
.lance-hero,
.lance-hero > div,
.lance-taskbar-wrap,
.lance-task-prompt-panel,
.lance-recommended-section {
  position: relative !important;
  z-index: 40 !important;
  isolation: isolate !important;
}

.lance-label-html,
.lance-label-html > div,
.lance-label-html .wrap,
.lance-prompt-label,
.lance-output-label,
.lance-section-label,
.lance-generation-label,
.lance-recommended-section .lance-section-label {
  position: relative !important;
  z-index: 60 !important;
}

.lance-main-row,
.lance-main-row > div,
.lance-panel,
.example-panel,
.prompt-examples,
.main-prompt-control,
.lance-output-panel,
.output-media-control,
.output-text-control,
.prompt-example-full-table {
  position: relative !important;
  z-index: 1 !important;
}

.lance-badges { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; margin: 4px auto 0; }
.lance-badges a { line-height: 0; }
.lance-badges img { height: 20px; width: auto; display: block; }
.lance-status, .lance-run-status { max-width: 1120px; margin: 8px auto !important; }
.lance-run-status p { margin: 0 !important; }
.lance-run-status-pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--lance-border); background: var(--lance-surface); color: var(--lance-text-muted); font-size: 14px; font-weight: 700; box-shadow: var(--lance-shadow); }
.lance-run-status-chip { width: 8px; height: 8px; border-radius: 999px; background: var(--lance-accent); box-shadow: 0 0 0 4px rgba(251,146,60,.18); }
.lance-run-status-dots i { display: inline-block; width: 4px; height: 4px; margin-left: 3px; border-radius: 999px; background: currentColor; opacity: .45; animation: lance-dot-pulse 1.1s infinite ease-in-out; }
.lance-run-status-dots i:nth-child(2) { animation-delay: .15s; }
.lance-run-status-dots i:nth-child(3) { animation-delay: .3s; }
@keyframes lance-dot-pulse { 40% { transform: translateY(-1px); opacity: 1; } }

.lance-main-row { display: grid !important; grid-template-columns: minmax(0, 1.16fr) minmax(0, 0.84fr) !important; gap: 18px !important; align-items: start !important; }
.lance-main-column { min-width: 0 !important; width: 100% !important; }
.lance-panel, .lance-control-field, .example-panel { border: 0 !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
.lance-panel > .form, .lance-control-field > .form, .lance-label-html, .lance-label-html > div, .lance-label-html .wrap { border: 0 !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important; min-height: 0 !important; }
.lance-section-label, .lance-generation-label { margin: 0 0 10px !important; font-weight: 800 !important; color: var(--body-text-color) !important; }
.lance-section-label { font-size: 18px !important; }
.lance-generation-label { font-size: 14px !important; }
.lance-label-icon { display: none !important; }
.lance-output-label { display: inline-flex !important; align-items: center !important; gap: 8px !important; }
.lance-output-label .lance-label-icon { display: inline-flex !important; align-items: center !important; justify-content: center !important; width: 20px !important; height: 20px !important; color: var(--lance-accent) !important; }
.lance-output-label .lance-label-icon svg { width: 18px !important; height: 18px !important; display: block !important; }

.lance-taskbar-wrap { max-width: 1120px; margin: 0 auto 12px !important; }
.task-selector {
  overflow-x: auto !important;
  padding: 4px 0 12px !important;
  scrollbar-width: thin;
  display: flex !important;
  justify-content: center !important;
}
.task-selector > .wrap, .task-selector .wrap {
  width: max-content !important;
  max-width: min(100%, 1080px) !important;
  margin: 0 auto !important;
  padding: 4px !important;
  display: flex !important;
  justify-content: center !important;
  flex-wrap: nowrap !important;
  gap: 10px !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.task-selector label {
  min-width: max-content !important;
  min-height: 38px !important;
  padding: 9px 18px !important;
  border: 0 !important;
  border-radius: 999px !important;
  background: #f1f5f9 !important;
  color: var(--lance-text-muted) !important;
  justify-content: center !important;
  white-space: nowrap !important;
}
.task-selector label:has(input:checked) { background: var(--lance-accent) !important; color: #0f172a !important; box-shadow: 0 6px 16px rgba(251,146,60,.22) !important; }
.task-selector input:checked + span { color: #0f172a !important; font-weight: 800 !important; }

/* Remove Gradio's outer task selector rectangle; keep only individual pills. */
.lance-taskbar-wrap,
.lance-taskbar-wrap > div,
.lance-taskbar-wrap > .form,
.lance-taskbar-wrap .block,
.task-selector,
.task-selector > div,
.task-selector > .form,
.task-selector .form,
.task-selector .wrap {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.task-selector > .wrap,
.task-selector .wrap {
  padding: 0 !important;
}
.task-selector label {
  background: #f8fafc !important;
  border: 1px solid rgba(148,163,184,.25) !important;
  box-shadow: 0 3px 10px rgba(15,23,42,.04) !important;
}
.task-selector label:has(input:checked) {
  background: var(--lance-accent) !important;
  border-color: transparent !important;
  color: #0f172a !important;
  box-shadow: 0 8px 18px rgba(249,115,22,.24) !important;
}
.task-selector input:checked + span { color: #0f172a !important; }

.lance-task-prompt-panel { max-width: 1040px; margin: 0 auto 10px !important; }
.main-prompt-control, .main-prompt-control > div, .main-prompt-control .wrap { border: 0 !important; background: transparent !important; box-shadow: none !important; }
.main-prompt-control textarea { min-height: 160px !important; padding: 18px !important; border: 1px solid var(--lance-border) !important; border-radius: 16px !important; background: var(--lance-surface) !important; color: var(--lance-text) !important; font-size: 15px !important; line-height: 1.45 !important; box-shadow: var(--lance-shadow) !important; }
.main-prompt-control textarea::placeholder { color: #94a3b8 !important; }
.prompt-options {
  position: relative !important;
  z-index: 2 !important;
  margin: 8px 0 16px !important;
  padding: 0 !important;
}
.prompt-options > .form {
  display: grid !important;
  grid-template-columns: repeat(4, max-content) !important;
  align-items: center !important;
  justify-content: start !important;
  justify-items: start !important;
  gap: 6px !important;
  width: max-content !important;
  max-width: 100% !important;
}
/* Keep only a single pill per control; remove the outer gray container look. */
.prompt-chip,
.prompt-chip > .form,
.prompt-chip > div,
.prompt-chip .block,
.prompt-chip .form,
.prompt-chip .container,
.prompt-chip .wrap {
  width: 100% !important;
  min-width: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.prompt-chip {
  display: block !important;
  min-width: 0 !important;
  width: auto !important;
  flex: 0 0 auto !important;
}
.prompt-chip .wrap,
.prompt-chip .container,
.prompt-chip > .form,
.prompt-chip .form {
  display: inline-flex !important;
  align-items: center !important;
  width: auto !important;
}
.prompt-chip button,
.prompt-chip [role="button"],
.prompt-chip select,
.prompt-chip input {
  width: auto !important;
  min-width: 58px !important;
  min-height: 32px !important;
  height: 32px !important;
  border-radius: 999px !important;
  border: 1px solid var(--lance-border) !important;
  outline: 0 !important;
  background: var(--lance-surface-muted) !important;
  color: var(--lance-text) !important;
  font-size: 10px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
  padding: 0 8px !important;
}
.video-resolution-row button,
.video-resolution-row [role="button"],
.video-resolution-row select,
.video-resolution-row input { min-width: 58px !important; }
.aspect-ratio-row button,
.aspect-ratio-row [role="button"],
.aspect-ratio-row select,
.aspect-ratio-row input { min-width: 48px !important; }
.video-duration-row button,
.video-duration-row [role="button"],
.video-duration-row select,
.video-duration-row input { min-width: 44px !important; }
.output-resolution-row button,
.output-resolution-row [role="button"],
.output-resolution-row select,
.output-resolution-row input { min-width: 70px !important; }
.prompt-chip button,
.prompt-chip [role="button"] { white-space: nowrap !important; }
.prompt-chip .icon-wrap,
.prompt-chip .select-arrow,
.prompt-chip .label-wrap,
.prompt-chip .block-title,
.prompt-chip .block-info,
.prompt-chip label {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
@media (max-width: 1200px) {
  .lance-main-row { grid-template-columns: minmax(0, 1.24fr) minmax(0, 0.76fr) !important; }
  .prompt-options > .form {
    grid-template-columns: repeat(4, max-content) !important;
    justify-content: start !important;
    gap: 4px !important;
  }
  .prompt-chip button, .prompt-chip [role="button"], .prompt-chip select, .prompt-chip input {
    font-size: 9.5px !important;
    min-width: 50px !important;
    padding: 0 6px !important;
  }
  .aspect-ratio-row button,
  .aspect-ratio-row [role="button"],
  .aspect-ratio-row select,
  .aspect-ratio-row input { min-width: 42px !important; }
  .video-duration-row button,
  .video-duration-row [role="button"],
  .video-duration-row select,
  .video-duration-row input { min-width: 40px !important; }
}

/* Compact left-aligned prompt parameter pills; not equal-width. */
.prompt-options {
  margin: 8px 0 16px !important;
  padding: 0 !important;
}
.prompt-options > .form {
  display: inline-flex !important;
  flex-wrap: nowrap !important;
  justify-content: flex-start !important;
  justify-items: start !important;
  align-items: center !important;
  gap: 6px !important;
  width: auto !important;
  max-width: 100% !important;
}
.prompt-chip,
.prompt-chip > .form,
.prompt-chip > div,
.prompt-chip .block,
.prompt-chip .form,
.prompt-chip .container,
.prompt-chip .wrap {
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
}
.prompt-chip button,
.prompt-chip [role="button"],
.prompt-chip select,
.prompt-chip input {
  width: auto !important;
  min-width: 0 !important;
  height: 30px !important;
  min-height: 30px !important;
  font-size: 9.5px !important;
  padding: 0 8px !important;
  border-radius: 999px !important;
}
.video-resolution-row button,
.video-resolution-row [role="button"],
.video-resolution-row select,
.video-resolution-row input { min-width: 50px !important; max-width: 58px !important; }
.aspect-ratio-row button,
.aspect-ratio-row [role="button"],
.aspect-ratio-row select,
.aspect-ratio-row input { min-width: 44px !important; max-width: 52px !important; }
.video-duration-row button,
.video-duration-row [role="button"],
.video-duration-row select,
.video-duration-row input { min-width: 38px !important; max-width: 46px !important; }
.output-resolution-row button,
.output-resolution-row [role="button"],
.output-resolution-row select,
.output-resolution-row input { min-width: 64px !important; max-width: 80px !important; }
@media (max-width: 1200px) {
  .prompt-options > .form {
    display: inline-flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    gap: 4px !important;
    width: auto !important;
  }
  .prompt-chip button,
  .prompt-chip [role="button"],
  .prompt-chip select,
  .prompt-chip input {
    font-size: 9px !important;
    padding: 0 6px !important;
    height: 29px !important;
    min-height: 29px !important;
  }
}

.lance-display-frame, .lance-display-frame > div, .lance-display-frame textarea, .output-media-control { width: 100% !important; }
.lance-output-panel { background: transparent !important; }
.lance-output-panel .lance-display-frame > div,
.lance-output-panel .lance-display-frame .wrap,
.lance-output-panel .output-media-control,
.lance-output-panel .output-media-control > div {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.lance-output-panel .output-media-control video,
.lance-output-panel .output-media-control img,
.lance-output-panel .lance-display-frame textarea {
  border-radius: 18px !important;
  border: 1px solid rgba(116, 126, 140, .34) !important;
  background: linear-gradient(180deg, rgba(250,251,253,.94), rgba(244,246,249,.9)) !important;
  box-shadow: 0 10px 28px rgba(15,23,42,.10), inset 0 0 0 1px rgba(255,255,255,.75) !important;
}
.lance-output-panel .lance-display-frame textarea { color: #101828 !important; }
.output-media-control video, .output-media-control img { border-radius: 18px !important; }
.lance-run-button { max-width: 1040px !important; margin: 10px auto 16px !important; border-radius: 12px !important; font-size: 18px !important; font-weight: 800 !important; }
button.lance-run-button, .lance-run-button button { width: 100% !important; border: 0 !important; border-radius: 12px !important; background: var(--lance-accent) !important; color: #0f172a !important; font-size: 18px !important; font-weight: 800 !important; box-shadow: 0 10px 24px rgba(249,115,22,.22) !important; }
button.lance-run-button:hover, .lance-run-button button:hover { background: var(--lance-accent-hover) !important; color: #0f172a !important; }

button.lance-run-button, .lance-run-button button {
  background: var(--lance-accent) !important;
  color: #0f172a !important;
  box-shadow: 0 10px 24px rgba(249,115,22,.22) !important;
}
button.lance-run-button:hover, .lance-run-button button:hover {
  background: var(--lance-accent-hover) !important;
  color: #0f172a !important;
}

.lance-advanced-accordion { max-width: 1040px; margin: 8px auto 0 !important; }
.lance-advanced-accordion .label-wrap, .lance-advanced-accordion summary { font-weight: 800 !important; }

.lance-recommended-section { max-width: 1040px; margin: 20px auto 0 !important; }
.lance-recommended-section .lance-section-label { text-align: left !important; font-size: 20px !important; margin-bottom: 12px !important; }
.prompt-example-full-table {
  max-height: 420px !important;
  overflow: auto !important;
  border: 1px solid rgba(148,163,184,.24) !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #ffffff, #f8fafc) !important;
  box-shadow: 0 12px 28px rgba(15,23,42,.07) !important;
  padding: 12px !important;
}
.prompt-example-full-table > .form { gap: 10px !important; }
.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button button {
  width: 100% !important;
  height: auto !important;
  min-height: 52px !important;
  max-height: 150px !important;
  padding: 12px 14px !important;
  border: 1px solid rgba(148,163,184,.22) !important;
  border-radius: 14px !important;
  background: #fff !important;
  color: var(--lance-text) !important;
  text-align: left !important;
  justify-content: flex-start !important;
  align-items: flex-start !important;
  white-space: normal !important;
  overflow-y: auto !important;
  box-shadow: 0 6px 16px rgba(15,23,42,.045) !important;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease !important;
}
.prompt-examples .prompt-example-row-button:hover,
.prompt-examples .prompt-example-row-button button:hover {
  transform: translateY(-1px) !important;
  border-color: rgba(251,146,60,.48) !important;
  box-shadow: 0 10px 22px rgba(15,23,42,.075) !important;
}
.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  line-height: 1.38 !important;
  color: var(--lance-text) !important;
}

.prompt-example-multimodal-row,
.prompt-example-multimodal-row > .form {
  width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  gap: 12px !important;
  align-items: stretch !important;
}
.prompt-example-multimodal-row > .form {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) 230px !important;
  padding: 8px !important;
  border: 1px solid rgba(148,163,184,.20) !important;
  border-radius: 16px !important;
  background: #fff !important;
  box-shadow: 0 6px 16px rgba(15,23,42,.045) !important;
}
.prompt-example-prompt-cell,
.prompt-example-prompt-cell > .form,
.prompt-example-media-cell,
.prompt-example-media-cell > .form {
  min-width: 0 !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.prompt-example-multimodal-row .prompt-example-row-button,
.prompt-example-multimodal-row .prompt-example-row-button button {
  height: 100% !important;
  min-height: 132px !important;
  max-height: 132px !important;
  border: 0 !important;
  box-shadow: none !important;
  background: #f8fafc !important;
}
.prompt-example-media-html,
.prompt-example-media-html > div,
.prompt-example-media-html .wrap {
  width: 100% !important;
  height: 132px !important;
  min-height: 132px !important;
  max-height: 132px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1px solid rgba(148,163,184,.22) !important;
  border-radius: 14px !important;
  background: #fff !important;
  box-shadow: none !important;
  overflow: hidden !important;
}
.prompt-example-media-html video,
.prompt-example-media-html img,
.example-preview-video,
.example-preview-image {
  width: 100% !important;
  height: 132px !important;
  border-radius: 12px !important;
  display: block !important;
  background: var(--lance-surface-muted) !important;
  object-fit: contain !important;
  object-position: center center !important;
}
.reference-media-fallback {
  width: 100% !important;
  height: 132px !important;
  border-radius: 12px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: var(--lance-surface-muted) !important;
  color: var(--lance-text-muted) !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  text-align: center !important;
}
@media (max-width: 760px) {
  .prompt-example-multimodal-row > .form { grid-template-columns: minmax(0, 1fr) 140px !important; }
  .prompt-example-multimodal-row .prompt-example-row-button,
  .prompt-example-multimodal-row .prompt-example-row-button button,
  .prompt-example-media-html,
  .prompt-example-media-html > div,
  .prompt-example-media-html .wrap,
  .prompt-example-media-html video,
  .prompt-example-media-html img,
  .example-preview-video,
  .example-preview-image {
    height: 108px !important;
    min-height: 108px !important;
    max-height: 108px !important;
  }
}

@media (max-width: 900px) { .lance-main-row { grid-template-columns: minmax(0, 1fr) !important; } .prompt-options { margin-top: 8px !important; } }

/* Fix recommended text case overlap: make each row taller and clip inside the card instead of overflowing into the next card. */
.prompt-example-full-table {
  max-height: none !important;
  overflow: visible !important;
  padding: 18px !important;
}
.prompt-example-full-table > .form {
  gap: 18px !important;
}
.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button button {
  min-height: 168px !important;
  height: auto !important;
  max-height: none !important;
  padding: 22px 24px !important;
  line-height: 1.62 !important;
  overflow: hidden !important;
  display: flex !important;
  align-items: flex-start !important;
}
.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
  line-height: 1.62 !important;
  overflow: hidden !important;
}
.prompt-example-multimodal-row .prompt-example-row-button,
.prompt-example-multimodal-row .prompt-example-row-button button,
.prompt-example-media-html,
.prompt-example-media-html > div,
.prompt-example-media-html .wrap,
.prompt-example-media-html video,
.prompt-example-media-html img,
.example-preview-video,
.example-preview-image,
.reference-media-fallback {
  min-height: 160px !important;
  height: 160px !important;
  max-height: 160px !important;
}

/* Larger text-only recommended case rows, especially Video Generation prompts.
   Avoid inner row scrolling; let each case card grow enough to show the prompt. */
.prompt-example-full-table {
  max-height: 560px !important;
}
.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button button {
  min-height: 96px !important;
  max-height: none !important;
  padding: 18px 20px !important;
  overflow-y: visible !important;
}
.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
  line-height: 1.55 !important;
}

/* Final UI polish: softer active task, looser examples, compact upper parameter row. */
.task-selector label:has(input:checked) {
  box-shadow: 0 4px 10px rgba(249,115,22,.12) !important;
}

.prompt-options {
  margin: 5px 0 14px !important;
}
.prompt-options > .form {
  gap: 7px !important;
}
.prompt-chip button,
.prompt-chip [role="button"],
.prompt-chip select,
.prompt-chip input {
  height: 31px !important;
  min-height: 31px !important;
  font-size: 10.5px !important;
  padding: 0 9px !important;
}
.video-resolution-row button,
.video-resolution-row [role="button"],
.video-resolution-row select,
.video-resolution-row input { min-width: 54px !important; max-width: 62px !important; }
.aspect-ratio-row button,
.aspect-ratio-row [role="button"],
.aspect-ratio-row select,
.aspect-ratio-row input { min-width: 48px !important; max-width: 56px !important; }
.video-duration-row button,
.video-duration-row [role="button"],
.video-duration-row select,
.video-duration-row input { min-width: 42px !important; max-width: 50px !important; }
.output-resolution-row button,
.output-resolution-row [role="button"],
.output-resolution-row select,
.output-resolution-row input { min-width: 68px !important; max-width: 86px !important; }

.lance-recommended-section { margin-top: 24px !important; }
.prompt-example-full-table {
  max-height: 480px !important;
  padding: 16px !important;
}
.prompt-example-full-table > .form {
  gap: 12px !important;
}
.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button button {
  min-height: 66px !important;
  padding: 16px 18px !important;
  line-height: 1.48 !important;
}
.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
  line-height: 1.48 !important;
}
.prompt-example-multimodal-row,
.prompt-example-multimodal-row > .form {
  gap: 14px !important;
}
.prompt-example-multimodal-row > .form {
  padding: 12px !important;
}
.prompt-example-multimodal-row .prompt-example-row-button,
.prompt-example-multimodal-row .prompt-example-row-button button,
.prompt-example-media-html,
.prompt-example-media-html > div,
.prompt-example-media-html .wrap,
.prompt-example-media-html video,
.prompt-example-media-html img,
.example-preview-video,
.example-preview-image,
.reference-media-fallback {
  min-height: 148px !important;
  height: 148px !important;
  max-height: 148px !important;
}

@media (max-width: 1200px) {
  .prompt-options { margin-top: 5px !important; }
  .prompt-chip button,
  .prompt-chip [role="button"],
  .prompt-chip select,
  .prompt-chip input {
    font-size: 10px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 7px !important;
  }
}

/* FINAL prompt display: show recommended text prompts completely.
   Cards grow with content; the page scrolls instead of clipping or inner-scrolling. */
.prompt-example-full-table,
.prompt-example-full-table > .form,
.prompt-examples,
.prompt-examples > .form {
  max-height: none !important;
  height: auto !important;
  overflow: visible !important;
}

.prompt-example-full-table {
  padding: 16px !important;
}

.prompt-example-full-table > .form {
  gap: 14px !important;
}

/* Text-only recommended cases: complete prompt display. */
.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button button {
  min-height: 96px !important;
  height: auto !important;
  max-height: none !important;
  padding: 18px 22px !important;
  overflow: visible !important;
  white-space: normal !important;
  display: block !important;
  text-align: left !important;
}

.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
  max-height: none !important;
  height: auto !important;
  overflow: visible !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: normal !important;
  line-height: 1.5 !important;
  text-overflow: unset !important;
  -webkit-line-clamp: unset !important;
  line-clamp: unset !important;
}

/* Multimodal rows keep fixed media height; only text-only prompt rows expand fully. */
.prompt-example-multimodal-row,
.prompt-example-multimodal-row > .form {
  max-height: none !important;
  overflow: visible !important;
  gap: 12px !important;
}

.prompt-example-multimodal-row > .form {
  padding: 12px !important;
}

.prompt-example-multimodal-row .prompt-example-row-button,
.prompt-example-multimodal-row .prompt-example-row-button button,
.prompt-example-media-html,
.prompt-example-media-html > div,
.prompt-example-media-html .wrap,
.prompt-example-media-html video,
.prompt-example-media-html img,
.example-preview-video,
.example-preview-image,
.reference-media-fallback {
  min-height: 148px !important;
  height: 148px !important;
  max-height: 148px !important;
}

/* FINAL output panel: show one visible border for empty Output Video/Image placeholders.
   The previous outer-layer removal made empty media outputs look borderless because
   the actual <video>/<img> element does not exist before generation. */
.lance-output-panel .output-media-control {
  min-height: 220px !important;
  border: 1px solid rgba(116,126,140,.34) !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, rgba(250,251,253,.94), rgba(244,246,249,.9)) !important;
  box-shadow: 0 10px 28px rgba(15,23,42,.10), inset 0 0 0 1px rgba(255,255,255,.75) !important;
  overflow: hidden !important;
}

.lance-output-panel .output-media-control > div,
.lance-output-panel .output-media-control .wrap {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.lance-output-panel .output-media-control video,
.lance-output-panel .output-media-control img {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  border-radius: 18px !important;
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
}

/* FINAL output text panel: match Output Video/Image with a single outer frame. */
.lance-output-panel .output-text-control {
  min-height: 220px !important;
  border: 1px solid rgba(116,126,140,.34) !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, rgba(250,251,253,.94), rgba(244,246,249,.9)) !important;
  box-shadow: 0 10px 28px rgba(15,23,42,.10), inset 0 0 0 1px rgba(255,255,255,.75) !important;
  overflow: hidden !important;
  padding: 0 !important;
}

.lance-output-panel .output-text-control > div,
.lance-output-panel .output-text-control .wrap,
.lance-output-panel .output-text-control .container {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.lance-output-panel .output-text-control textarea {
  min-height: 220px !important;
  border: 0 !important;
  border-radius: 18px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #101828 !important;
  padding: 18px !important;
  resize: none !important;
}

/* FINAL parameter sizing: one size larger while staying on one row. */
.prompt-options > .form {
  display: inline-flex !important;
  flex-wrap: nowrap !important;
  justify-content: flex-start !important;
  align-items: center !important;
  gap: 8px !important;
  width: auto !important;
  max-width: 100% !important;
}

.prompt-chip button,
.prompt-chip [role="button"],
.prompt-chip select,
.prompt-chip input {
  height: 36px !important;
  min-height: 36px !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  padding-left: 12px !important;
  padding-right: 12px !important;
}

.video-resolution-row button,
.video-resolution-row [role="button"],
.video-resolution-row select,
.video-resolution-row input {
  min-width: 74px !important;
  max-width: 84px !important;
}

.aspect-ratio-row button,
.aspect-ratio-row [role="button"],
.aspect-ratio-row select,
.aspect-ratio-row input {
  min-width: 72px !important;
  max-width: 82px !important;
}

.video-duration-row button,
.video-duration-row [role="button"],
.video-duration-row select,
.video-duration-row input {
  min-width: 62px !important;
  max-width: 72px !important;
}

.output-resolution-row button,
.output-resolution-row [role="button"],
.output-resolution-row select,
.output-resolution-row input {
  min-width: 92px !important;
  max-width: 114px !important;
}

@media (max-width: 1200px) {
  .prompt-options > .form {
    gap: 6px !important;
  }
  .prompt-chip button,
  .prompt-chip [role="button"],
  .prompt-chip select,
  .prompt-chip input {
    height: 34px !important;
    min-height: 34px !important;
    font-size: 11px !important;
    padding-left: 9px !important;
    padding-right: 9px !important;
  }
  .video-resolution-row button,
  .video-resolution-row [role="button"],
  .video-resolution-row select,
  .video-resolution-row input {
    min-width: 66px !important;
    max-width: 76px !important;
  }
  .aspect-ratio-row button,
  .aspect-ratio-row [role="button"],
  .aspect-ratio-row select,
  .aspect-ratio-row input {
    min-width: 64px !important;
    max-width: 74px !important;
  }
  .video-duration-row button,
  .video-duration-row [role="button"],
  .video-duration-row select,
  .video-duration-row input {
    min-width: 56px !important;
    max-width: 66px !important;
  }
}

.lance-run-button {
  margin-bottom: 6px !important;
}

/* FINAL: frame interpolation is disabled and must not be visible. */

/* FINAL label visibility fix:
   Gradio can keep dark-theme text variables even when this app forces a white
   surface, so the labels looked like they were under a white layer. Force the
   specific section labels to be fully opaque, above panels, and dark on white. */
.lance-label-html,
.lance-label-html > div,
.lance-label-html .wrap,
.lance-label-html .prose,
.lance-prompt-label,
.lance-output-label,
.lance-section-label,
.lance-generation-label,
.lance-recommended-section,
.lance-recommended-section .lance-section-label {
  position: relative !important;
  z-index: 9999 !important;
  opacity: 1 !important;
  visibility: visible !important;
  mix-blend-mode: normal !important;
  filter: none !important;
  color: var(--lance-text) !important;
  text-shadow: none !important;
  background: transparent !important;
  transform: translateZ(0) !important;
}

.lance-prompt-label,
.lance-output-label,
.lance-recommended-section .lance-section-label {
  display: inline-flex !important;
  align-items: center !important;
  min-height: 24px !important;
  line-height: 1.25 !important;
  font-weight: 800 !important;
}

.lance-output-label .lance-label-icon {
  color: var(--lance-accent) !important;
  opacity: 1 !important;
}

/* Keep the white cards below the label layer, including Gradio internal wrappers. */
.lance-main-row,
.lance-main-row > div,
.lance-panel,
.lance-panel > .form,
.main-prompt-control,
.main-prompt-control > div,
.main-prompt-control .wrap,
.lance-output-panel,
.lance-output-panel > div,
.lance-display-frame,
.output-media-control,
.output-text-control,
.example-panel,
.prompt-examples,
.prompt-example-full-table {
  position: relative !important;
  z-index: 1 !important;
}


/* FINAL output-title visibility fix:
   Explicitly target the dynamic Output Video / Output Image / Output Text HTML label.
   These labels are updated by Python, so use stable wrapper classes rather than text content. */
.lance-output-column,
.lance-output-column .lance-output-panel {
  position: relative !important;
  z-index: 20 !important;
  overflow: visible !important;
  isolation: isolate !important;
}

.lance-output-column .lance-output-panel .lance-label-html,
.lance-output-column .lance-output-panel .lance-label-html > div,
.lance-output-column .lance-output-panel .lance-label-html .wrap,
.lance-output-column .lance-output-panel .lance-label-html .prose,
.lance-output-column .lance-output-panel .lance-output-label,
.lance-output-column .lance-output-panel .lance-output-label *,
.lance-output-column .lance-output-panel .lance-output-label-text {
  position: relative !important;
  z-index: 2147483647 !important;
  opacity: 1 !important;
  visibility: visible !important;
  mix-blend-mode: normal !important;
  filter: none !important;
  color: #111827 !important;
  background: transparent !important;
  text-shadow: none !important;
  transform: translateZ(0) !important;
}

.lance-output-column .lance-output-panel .lance-label-html {
  display: block !important;
  min-height: 26px !important;
  height: auto !important;
  margin: 0 0 10px 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  pointer-events: none !important;
}

.lance-output-column .lance-output-panel .lance-output-label {
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  min-height: 24px !important;
  line-height: 1.25 !important;
  font-weight: 800 !important;
  color: #111827 !important;
}

.lance-output-column .lance-output-panel .lance-output-label .lance-label-icon,
.lance-output-column .lance-output-panel .lance-output-label .lance-label-icon *,
.lance-output-column .lance-output-panel .lance-output-label svg {
  color: var(--lance-accent) !important;
  stroke: currentColor !important;
  opacity: 1 !important;
  visibility: visible !important;
}

.lance-output-column .lance-output-panel .output-media-control,
.lance-output-column .lance-output-panel .output-text-control,
.lance-output-column .lance-output-panel .lance-display-frame,
.lance-output-column .lance-output-panel .lance-display-frame > div,
.lance-output-column .lance-output-panel .lance-display-frame .wrap {
  position: relative !important;
  z-index: 0 !important;
}


"""