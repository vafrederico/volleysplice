package com.volleycut.nativeanalysis;

import java.util.LinkedHashMap;
import java.util.Map;

final class NanoProfiler {
    private final LinkedHashMap<String, Long> nanos = new LinkedHashMap<>();

    void add(String name, long elapsedNanos) {
        nanos.merge(name, Math.max(0, elapsedNanos), Long::sum);
    }

    Map<String, Double> milliseconds() {
        LinkedHashMap<String, Double> result = new LinkedHashMap<>();
        for (Map.Entry<String, Long> entry : nanos.entrySet()) {
            result.put(entry.getKey(), entry.getValue() / 1_000_000.0);
        }
        return result;
    }

    void appendMilliseconds(String prefix, Map<String, Double> source) {
        for (Map.Entry<String, Double> entry : source.entrySet()) {
            nanos.put(prefix + entry.getKey(), Math.round(entry.getValue() * 1_000_000));
        }
    }
}
