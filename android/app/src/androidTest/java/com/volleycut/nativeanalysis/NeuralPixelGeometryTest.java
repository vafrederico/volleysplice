package com.volleycut.nativeanalysis;

import android.graphics.ImageFormat;
import android.media.Image;
import android.media.ImageReader;
import android.media.ImageWriter;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.volleycut.neural.NeuralVideoEncoder;
import com.volleycut.video.YuvColorConversion;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.nio.ByteBuffer;
import static org.junit.Assert.*;

/** An asymmetric real YUV Image tests orientation independently of decoder parity. */
@RunWith(AndroidJUnit4.class)
public final class NeuralPixelGeometryTest {
    @Test public void rotatedPixelsMatchExplicitClockwiseCoordinateMapping() throws Exception {
        final int size=224, plane=size*size;
        try(ImageReader reader=ImageReader.newInstance(size,size,ImageFormat.YUV_420_888,2);
            ImageWriter writer=ImageWriter.newInstance(reader.getSurface(),2);
            Image image=writer.dequeueInputImage()) {
            Image.Plane[] source=image.getPlanes();
            for(int c=0;c<3;c++) {
                ByteBuffer values=source[c].getBuffer();
                int dimension=c==0?size:size/2;
                for(int y=0;y<dimension;y++) for(int x=0;x<dimension;x++) {
                    int value=c==0?32+x/2+y/4:c==1?96+x%8:160+y%8;
                    values.put(y*source[c].getRowStride()+x*source[c].getPixelStride(),(byte)value);
                }
            }
            double[] roi={0,0,1,1};
            YuvColorConversion color=YuvColorConversion.of(YuvColorConversion.BT709,YuvColorConversion.LIMITED);
            float[] baseline=new float[plane*3];
            NeuralVideoEncoder.prepare(image,baseline,size,roi,color,0);
            for(int rotation:new int[]{90,180,270}) {
                float[] actual=new float[plane*3];
                NeuralVideoEncoder.prepare(image,actual,size,roi,color,rotation);
                for(int y=0;y<size;y++) for(int x=0;x<size;x++) {
                    int sx=rotation==90?y:rotation==180?size-1-x:size-1-y;
                    int sy=rotation==90?size-1-x:rotation==180?size-1-y:x;
                    for(int c=0;c<3;c++) assertEquals(baseline[c*plane+sy*size+sx],actual[c*plane+y*size+x],0);
                }
            }
        }
    }
}
