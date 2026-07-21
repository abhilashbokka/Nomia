// Keyboard dispatcher for the review grid: Space/Enter confirm, E edit name, S skip, ↑/↓
// navigate. Inert whenever a text input/select has focus, so it never fights normal typing
// in the left panel or an in-progress name edit.

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

export function setupKeyboardHandler(handlers) {
  document.addEventListener("keydown", (event) => {
    if (isTypingTarget(document.activeElement)) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    switch (event.key) {
      case " ":
      case "Enter":
        event.preventDefault();
        handlers.onConfirm();
        break;
      case "e":
      case "E":
        event.preventDefault();
        handlers.onEdit();
        break;
      case "s":
      case "S":
        event.preventDefault();
        handlers.onSkip();
        break;
      case "ArrowDown":
        event.preventDefault();
        handlers.onNavigate(1);
        break;
      case "ArrowUp":
        event.preventDefault();
        handlers.onNavigate(-1);
        break;
      default:
        break;
    }
  });
}
