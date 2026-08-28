import assert from "node:assert/strict";
import test from "node:test";

import { scrollElementIntoContainer } from "../../prod/src/lib/scroll-container.ts";

type ScrollBoxOptions = {
  clientHeight?: number;
  clientTop?: number;
  scrollTop?: number;
  top?: number;
};

function scrollBox(options: ScrollBoxOptions = {}) {
  const calls: ScrollToOptions[] = [];
  const top = options.top ?? 100;
  const clientHeight = options.clientHeight ?? 200;
  const element = {
    clientHeight,
    clientTop: options.clientTop ?? 0,
    scrollTop: options.scrollTop ?? 0,
    getBoundingClientRect: () => ({ top, bottom: top + clientHeight }),
    scrollTo: (scrollOptions: ScrollToOptions) => calls.push(scrollOptions),
  } as unknown as HTMLElement;
  return { calls, element };
}

function row(top: number, bottom: number): HTMLElement {
  return {
    getBoundingClientRect: () => ({ top, bottom }),
  } as unknown as HTMLElement;
}

test("does not scroll when the selected row is already visible", () => {
  const container = scrollBox({ scrollTop: 80 });

  scrollElementIntoContainer(container.element, row(140, 180));

  assert.deepEqual(container.calls, []);
});

test("scrolls only the container far enough to reveal a row below it", () => {
  const container = scrollBox({ scrollTop: 80 });

  scrollElementIntoContainer(container.element, row(270, 340));

  assert.deepEqual(container.calls, [{ top: 120, behavior: "smooth" }]);
});

test("scrolls only the container far enough to reveal a row above it", () => {
  const container = scrollBox({ clientTop: 2, scrollTop: 80 });

  scrollElementIntoContainer(container.element, row(72, 112));

  assert.deepEqual(container.calls, [{ top: 50, behavior: "smooth" }]);
});

test("does not try to scroll a hidden container", () => {
  const container = scrollBox({ clientHeight: 0 });

  scrollElementIntoContainer(container.element, row(300, 340));

  assert.deepEqual(container.calls, []);
});
