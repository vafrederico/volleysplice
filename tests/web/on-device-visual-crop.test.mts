import assert from "node:assert/strict";
import test from "node:test";
import { visualCrop } from "../../prod/src/lib/on-device/visual-crop.ts";

test("visual crop rounds half-pixel coordinates to even like the training extractor", () => {
  assert.deepEqual(visualCrop({x:0.25,y:0.35,width:0.4,height:0.2},10,10),
    {left:2,top:4,width:4,height:2});
});

test("visual crop clips edges and preserves one source pixel for degenerate edge selections", () => {
  assert.deepEqual(visualCrop({x:1,y:1,width:0,height:0},1920,1080),
    {left:1919,top:1079,width:1,height:1});
  assert.deepEqual(visualCrop({x:0,y:0,width:1,height:1},1920,1080),
    {left:0,top:0,width:1920,height:1080});
});
