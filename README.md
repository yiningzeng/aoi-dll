aoi-api
# 一定要x64 release模式编译
# demo
```c#
           List<string> fileList = new List<string>();

            for(int i =0; i<119; i++)
            {
                fileList.Add(@"I:\windows_software\0\F" + i + ".jpg");
            }

            int n_rows = 7;
            int n_cols = 17;

            double or_hl = 0.2; // lower bound for horizontal overlap ratio
            double or_hu = 0.25; // upper
            double or_vl = 0.05; // vertical
            double or_vu = 0.1;
            double dr_hu = 0.01; // upper bound for horizontal drift ratio
            double dr_vu = 0.01; // 

            Mat dst = null;
            Rectangle roi0 = new Rectangle(); //上一行第一张的区域
                            // first row 
            Rectangle roi = new Rectangle(); // 左对齐的参考的区域
             // first row 
            for (int col = 0; col < n_cols; ++col)
            {
                Mat img = new Mat(fileList[col], Emgu.CV.CvEnum.LoadImageType.AnyColor);
                if (col == 0)
                {
                    roi0 = new Rectangle(Convert.ToInt32(img.Cols * (n_cols - 1) * dr_hu), Convert.ToInt32(img.Rows * (n_rows - 1) * dr_vu), img.Cols, img.Rows);
                    dst = new Mat(Convert.ToInt32(img.Rows * (n_rows + (n_rows - 1) * (dr_vu * 2 - or_vl))), Convert.ToInt32(img.Cols * (n_cols + (n_cols - 1) * (dr_hu * 2 - or_hl))), img.Depth, 3); // 第一张图不要0,0 最好留一些像素
                    roi = roi0;
                }
                else
                {
                    AoiAi.stitchv2(dst.Ptr, roi, img.Ptr, ref roi, (int)AoiAi.side.left, Convert.ToInt32(img.Cols * or_hl), Convert.ToInt32(img.Cols * or_hu), Convert.ToInt32(img.Rows * dr_vu));
                }

                AoiAi.copy_to(dst.Ptr, img.Ptr, roi);
                //AoiAi.addPatch(dst.Ptr, img.Ptr, roi.X, roi.Y);
                #region 这里去掉
                CvInvoke.NamedWindow("AJpg", NamedWindowType.Normal); //创建一个显示窗口
                CvInvoke.Imshow("AJpg", dst);
                
                char key = (char)CvInvoke.WaitKey(1);
                if (key == 0x1b || key == 'q') continue;
                #endregion 这里去掉

            }

            // other rows
            for (int row = 1; row < n_rows; ++row)
            {
                for (int col = 0; col < n_cols; ++col)
                {
                    Mat img = new Mat(fileList[n_cols * row + col], Emgu.CV.CvEnum.LoadImageType.AnyColor);
                    //std::cout << n_cols * row + col << "\n";

                    if (col == 0)
                    {
                        AoiAi.stitchv2(dst.Ptr, roi0, img.Ptr, ref roi0, (int)AoiAi.side.up, Convert.ToInt32(img.Cols * or_vl), Convert.ToInt32(img.Cols * or_vu), Convert.ToInt32(img.Rows * dr_hu));
                        roi = roi0;
                    }
                    else
                    {
                        AoiAi.stitchv2(dst.Ptr, roi, img.Ptr, ref roi, (int)AoiAi.side.left, Convert.ToInt32(img.Cols * or_hl), Convert.ToInt32(img.Cols * or_hu), Convert.ToInt32(img.Rows * dr_vu), (int)AoiAi.side.up, Convert.ToInt32(img.Rows * or_vl), Convert.ToInt32(img.Rows * or_vu), Convert.ToInt32(img.Cols * dr_hu));
                    }
                    AoiAi.copy_to(dst.Ptr, img.Ptr, roi);
                    #region 这里去掉
                    CvInvoke.NamedWindow("AJpg", NamedWindowType.Normal); //创建一个显示窗口
                    CvInvoke.Imshow("AJpg", dst);
                    char key = (char)CvInvoke.WaitKey(1);
                    if (key == 0x1b || key == 'q') continue;
                    #endregion 这里去掉
                }
            }
```