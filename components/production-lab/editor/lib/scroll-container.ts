export function scrollElementIntoContainer(
  container: HTMLElement,
  element: HTMLElement,
  behavior: ScrollBehavior = "smooth",
): void {
  if (container.clientHeight <= 0) return;

  const containerRect = container.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();
  const visibleTop = containerRect.top + container.clientTop;
  const visibleBottom = visibleTop + container.clientHeight;
  let nextScrollTop: number | null = null;

  if (elementRect.top < visibleTop) {
    nextScrollTop = container.scrollTop + elementRect.top - visibleTop;
  } else if (elementRect.bottom > visibleBottom) {
    nextScrollTop = container.scrollTop + elementRect.bottom - visibleBottom;
  }

  if (nextScrollTop === null) return;
  container.scrollTo({
    top: Math.max(0, nextScrollTop),
    behavior,
  });
}
