package com.volleycut.neural;

/** Fractional 7x7 feature-cell overlap with the four content-relative regions. */
public final class RegionalPoolWeights {
    private RegionalPoolWeights() {}
    public static float[] forGeometry(int width, int height, int rotation, double[] roi) {
        if (width <= 0 || height <= 0 || (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270)) {
            throw new IllegalArgumentException("Invalid neural image geometry");
        }
        if (rotation == 90 || rotation == 270) { int swap=width; width=height; height=swap; }
        int croppedWidth=(int)Math.round(width*roi[2]), croppedHeight=(int)Math.round(height*roi[3]);
        if (croppedWidth <= 0 || croppedHeight <= 0) throw new IllegalArgumentException("Empty neural crop");
        double scale=Math.min(224.0/croppedWidth,224.0/croppedHeight);
        int w=(int)Math.round(croppedWidth*scale), h=(int)Math.round(croppedHeight*scale);
        double left=((224-w)/2)/224.0, top=((224-h)/2)/224.0;
        double right=left+w/224.0, bottom=top+h/224.0;
        double[][] regions={{0,1},{.5,1},{0,.5},{.4,.6}};
        float[] weights=new float[196];
        for(int region=0;region<4;region++) {
            double a=top+regions[region][0]*(bottom-top), b=top+regions[region][1]*(bottom-top);
            double[] area=new double[49]; double total=0;
            for(int y=0;y<7;y++) for(int x=0;x<7;x++) {
                double v=Math.max(0,Math.min((x+1)/7.0,right)-Math.max(x/7.0,left))
                    *Math.max(0,Math.min((y+1)/7.0,b)-Math.max(y/7.0,a));
                area[y*7+x]=v; total+=v;
            }
            if(total<=0) throw new IllegalArgumentException("Empty regional pool");
            for(int i=0;i<49;i++) weights[region*49+i]=(float)(area[i]/total);
        }
        return weights;
    }
}
