// ocr: macOS Vision text recognition, accurate mode, no autocorrect.
// usage: ocr IMAGE... -> one JSON line per image:
//   {"path": ..., "lines": [{"t": text, "c": confidence, "a": [alternate readings], "b": [x, y, w, h]}]}
// b is normalised, origin bottom-left (Vision's convention). a lists the other top-5 readings.
import Foundation
import Vision
import AppKit

for path in CommandLine.arguments.dropFirst() {
    var lines: [[String: Any]] = []
    if let img = NSImage(contentsOfFile: path),
       let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        try? VNImageRequestHandler(cgImage: cg).perform([req])
        for obs in (req.results ?? []) {
            let cands = obs.topCandidates(5)
            if let t = cands.first {
                let bb = obs.boundingBox
                lines.append(["t": t.string, "c": Double(t.confidence),
                              "a": cands.dropFirst().map { $0.string },
                              "b": [bb.origin.x, bb.origin.y, bb.size.width, bb.size.height]])
            }
        }
    }
    let out: [String: Any] = ["path": path, "lines": lines]
    if let data = try? JSONSerialization.data(withJSONObject: out),
       let s = String(data: data, encoding: .utf8) { print(s) }
}
