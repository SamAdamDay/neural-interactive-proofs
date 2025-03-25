import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@${mermaid_version}/dist/mermaid.esm.min.mjs";
import elkLayouts from "https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@${elk_version}/dist/mermaid-layout-elk.esm.min.mjs";

mermaid.registerLayoutLoaders(elkLayouts);

/**
 * Save the sources of the mermaid diagrams in the 'data-source' attribute
 * of the element to be able to re-render them when the theme changes.
 */
function saveMermaidSources() {
  document.querySelectorAll(".mermaid").forEach((element) => {
    element.setAttribute("data-source", element.textContent);
  });
}

/**
 * Determine the theme based on the user's preference and initialize Mermaid with it.
 *
 */
function determineThemeAndInitialize() {
  const theme =
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "default";
  mermaid.initialize({
    startOnLoad: false,
    theme: theme,
  });
}

/**
 * Determine the theme based on the user's preference and re-render the Mermaid diagrams.
 */
function determineThemeAndReRender() {
  determineThemeAndInitialize();
  document.querySelectorAll(".mermaid").forEach((element) => {
    element.removeAttribute("data-processed");
    element.textContent = element.getAttribute("data-source");
  });
  mermaid.run();
}

determineThemeAndInitialize();

window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", ({ matches }) => {
    determineThemeAndReRender();
  });

// Save the sources of the mermaid diagrams to be able to re-render them when the theme
// changes. A bit of a hack, and relies on this being called before `mermaid.run()` gets
// called, which should happen because this script is loaded before the mermaid script.
window.addEventListener("load", () => saveMermaidSources());
