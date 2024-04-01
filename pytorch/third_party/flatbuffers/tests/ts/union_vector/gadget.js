// automatically generated by the FlatBuffers compiler, do not modify
import { FallingTub } from './falling-tub.js';
import { HandFan } from './hand-fan.js';
export var Gadget;
(function (Gadget) {
    Gadget[Gadget["NONE"] = 0] = "NONE";
    Gadget[Gadget["FallingTub"] = 1] = "FallingTub";
    Gadget[Gadget["HandFan"] = 2] = "HandFan";
})(Gadget = Gadget || (Gadget = {}));
export function unionToGadget(type, accessor) {
    switch (Gadget[type]) {
        case 'NONE': return null;
        case 'FallingTub': return accessor(new FallingTub());
        case 'HandFan': return accessor(new HandFan());
        default: return null;
    }
}
export function unionListToGadget(type, accessor, index) {
    switch (Gadget[type]) {
        case 'NONE': return null;
        case 'FallingTub': return accessor(index, new FallingTub());
        case 'HandFan': return accessor(index, new HandFan());
        default: return null;
    }
}
