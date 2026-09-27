#include <jni.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
void invalid(JNIEnv* env, const char* message) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), message);
}

struct Plane {
    const uint8_t* data;
    int row, pixel;
    int at(int x, int y) const { return data[int64_t(y) * row + int64_t(x) * pixel]; }
};

bool plane(JNIEnv* env, jobject buffer, int row, int pixel, int limit,
           int lastX, int lastY, Plane& out) {
    if (!buffer) { invalid(env, "Missing YUV plane"); return false; }
    auto* data = static_cast<const uint8_t*>(env->GetDirectBufferAddress(buffer));
    const jlong capacity = env->GetDirectBufferCapacity(buffer);
    const int64_t last = int64_t(lastY) * row + int64_t(lastX) * pixel;
    if (!data || row <= 0 || pixel <= 0 || limit < 0 || capacity < limit || last >= limit) {
        invalid(env, "YUV plane does not contain the complete crop within its limit");
        return false;
    }
    out = {data, row, pixel};
    return true;
}

struct Color {
    jint table[1280];
    static int channel(int value) {
        return std::clamp((value + 32768) >> 16, 0, 255);
    }
    void rgb(int y, int u, int v, int& r, int& g, int& b) const {
        const int l = table[y];
        r = channel(l + table[256 + v]);
        g = channel(l + table[512 + u] + table[768 + v]);
        b = channel(l + table[1024 + u]);
    }
};

int roundEven(double value) {
    const int low = static_cast<int>(std::floor(value));
    const double fraction = value - low;
    return low + (fraction > .5 || (fraction == .5 && (low & 1)));
}

struct Output {
    uint8_t* data;
    int width, height, rotation;
    void put(int x, int y, double r, double g, double b) const {
        int dx = x, dy = y;
        switch (rotation) {
            case 90: dx = width - 1 - y; dy = x; break;
            case 180: dx = width - 1 - x; dy = height - 1 - y; break;
            case 270: dx = y; dy = height - 1 - x; break;
        }
        auto* pixel = data + (int64_t(dy) * width + dx) * 4;
        pixel[0] = roundEven(r); pixel[1] = roundEven(g); pixel[2] = roundEven(b); pixel[3] = 255;
    }
};

struct Span { int first; std::vector<double> weights; };
std::vector<Span> spans(int source, int target) {
    std::vector<Span> result(target);
    const double scale = double(source) / target;
    for (int i = 0; i < target; ++i) {
        const double start = i * scale, end = std::min(double(source), (i + 1) * scale);
        Span& span = result[i];
        span.first = static_cast<int>(std::floor(start));
        const int last = static_cast<int>(std::ceil(end));
        span.weights.resize(last - span.first);
        for (int k = 0; k < last - span.first; ++k) {
            const int pixel = span.first + k;
            span.weights[k] = std::max(0.0, std::min(end, double(pixel + 1)) - std::max(start, double(pixel)));
        }
    }
    return result;
}

void integerArea(const Plane& y, const Plane& u, const Plane& v, const Color& color,
                 const Output& output, int left, int top, int width, int height, int rw, int rh) {
    const int scaleX = width / rw, scaleY = height / rh;
    const double denominator = double(scaleX) * scaleY;
    for (int dy = 0; dy < rh; ++dy) {
        for (int dx = 0; dx < rw; ++dx) {
            int64_t red = 0, green = 0, blue = 0;
            const int x0 = left + dx * scaleX, y0 = top + dy * scaleY;
            for (int sy = y0; sy < y0 + scaleY; ++sy) {
                for (int sx = x0; sx < x0 + scaleX; ++sx) {
                    int r, g, b;
                    color.rgb(y.at(sx, sy), u.at(sx / 2, sy / 2), v.at(sx / 2, sy / 2), r, g, b);
                    red += r; green += g; blue += b;
                }
            }
            output.put(dx, dy, red / denominator, green / denominator, blue / denominator);
        }
    }
}

void weightedArea(const Plane& y, const Plane& u, const Plane& v, const Color& color,
                  const Output& output, int left, int top, int width, int height, int rw, int rh) {
    const auto columns = spans(width, rw), rows = spans(height, rh);
    const double denominator = (double(width) / rw) * (double(height) / rh);
    for (int dy = 0; dy < rh; ++dy) for (int dx = 0; dx < rw; ++dx) {
        double red = 0, green = 0, blue = 0;
        const auto& row = rows[dy]; const auto& col = columns[dx];
        for (size_t iy = 0; iy < row.weights.size(); ++iy) {
            const int sy = top + row.first + static_cast<int>(iy);
            for (size_t ix = 0; ix < col.weights.size(); ++ix) {
                const int sx = left + col.first + static_cast<int>(ix);
                int r, g, b;
                color.rgb(y.at(sx, sy), u.at(sx / 2, sy / 2), v.at(sx / 2, sy / 2), r, g, b);
                const double weight = row.weights[iy] * col.weights[ix];
                red += r * weight; green += g * weight; blue += b * weight;
            }
        }
        output.put(dx, dy, red / denominator, green / denominator, blue / denominator);
    }
}
} // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_volleycut_video_NativeYuvArea_resample(JNIEnv* env, jclass,
        jobject yBuffer, jint yRow, jint yPixel, jint yLimit,
        jobject uBuffer, jint uRow, jint uPixel, jint uLimit,
        jobject vBuffer, jint vRow, jint vPixel, jint vLimit,
        jintArray geometry, jintArray tables, jbyteArray rgba) {
    if (!geometry || !tables || !rgba || env->GetArrayLength(geometry) != 9
            || env->GetArrayLength(tables) != 1280) {
        invalid(env, "Invalid area conversion plan"); return;
    }
    jint g[9]; env->GetIntArrayRegion(geometry, 0, 9, g);
    const int left = g[0], top = g[1], width = g[2], height = g[3], rw = g[4], rh = g[5];
    const int rotation = g[6], ow = g[7], oh = g[8];
    const bool transpose = rotation == 90 || rotation == 270;
    if (left < 0 || top < 0 || width <= 0 || height <= 0 || rw <= 0 || rh <= 0
            || ow <= 0 || oh <= 0 || int64_t(left) + width > std::numeric_limits<int>::max()
            || int64_t(top) + height > std::numeric_limits<int>::max()
            || rotation < 0 || rotation >= 360 || rotation % 90 != 0
            || rw != (transpose ? oh : ow) || rh != (transpose ? ow : oh)
            || int64_t(ow) * oh > std::numeric_limits<int>::max() / 4
            || int64_t(ow) * oh * 4 != env->GetArrayLength(rgba)) {
        invalid(env, "Invalid area conversion geometry"); return;
    }
    Plane y{}, u{}, v{};
    const int lastX = left + width - 1, lastY = top + height - 1;
    if (!plane(env, yBuffer, yRow, yPixel, yLimit, lastX, lastY, y)
            || !plane(env, uBuffer, uRow, uPixel, uLimit, lastX / 2, lastY / 2, u)
            || !plane(env, vBuffer, vRow, vPixel, vLimit, lastX / 2, lastY / 2, v)) return;
    Color color{}; env->GetIntArrayRegion(tables, 0, 1280, color.table);
    if (env->ExceptionCheck()) return;
    jbyte* bytes = env->GetByteArrayElements(rgba, nullptr);
    if (!bytes) return;
    Output output{reinterpret_cast<uint8_t*>(bytes), ow, oh, rotation};
    if (width % rw == 0 && height % rh == 0) {
        integerArea(y, u, v, color, output, left, top, width, height, rw, rh);
    } else {
        weightedArea(y, u, v, color, output, left, top, width, height, rw, rh);
    }
    env->ReleaseByteArrayElements(rgba, bytes, 0);
}
