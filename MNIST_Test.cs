using System;
using T_SNE;
using System.IO;
using Accord.Math;
using System.Diagnostics;

namespace MainTest
{
    class Program
    {

        static void Main(string[] args)
        {

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();


            String[] lines_X = File.ReadAllLines(@"C:\Users\v-jiehu\source\repos\T-SNE\T-SNE\data\mnist2500_X.txt");
            String[] lines_Y = File.ReadAllLines(@"C:\Users\v-jiehu\source\repos\T-SNE\T-SNE\data\mnist2500_labels.txt");
            // var lines_X = input_X.Trim().Split('\n');
            // var line1_X = lines_X[0].Split(new string[] { "   " }, StringSplitOptions.None);

            var m = Matrix.Create(lines_X.GetLength(0), lines_X[0].Split().GetLength(0), 0.0);
            var r = 0;
            var c = 0;

            foreach (var line in lines_X)
            {

                foreach (var w in line.Trim().Split(new string[] { "   " }, StringSplitOptions.None))
                {
                    try
                    {
                        m[r, c] = Convert.ToDouble(w);
                    }
                    catch (FormatException)
                    {
                        Console.WriteLine("Unable to convert '{0}' to a Double. - Data", w);
                        Console.WriteLine(r);
                        Console.WriteLine(c);
                    }
                    catch (OverflowException)
                    {
                        Console.WriteLine("'{0}' is outside the range of a Double.", w);
                    }

                    c++;
                }
                r++;
                c = 0;
            }

            //var lines_Y = input_labels.Trim().Split('\n');
            var labels = Vector.Create(lines_Y.GetLength(0), 0.0);
            c = 0;

            foreach (var w in lines_Y)
            {

                try
                {
                    labels[c] = Convert.ToDouble(w);
                }
                catch (FormatException)
                {
                    Console.WriteLine("Unable to convert '{0}' to a Double. - Label", w);
                }
                catch (OverflowException)
                {
                    Console.WriteLine("'{0}' is outside the range of a Double.", w);
                }

                c++;

            }


            stopWatch.Stop();
            // Get the elapsed time as a TimeSpan value.
            TimeSpan ts = stopWatch.Elapsed;
            Console.WriteLine("Reading files takes: " + ts.Seconds.ToString() + " seconds");

            // T-SNE starts
            stopWatch.Restart();
            var Y = new TSNE(m, 2, 2, 30.0)._TSNE();
            
            stopWatch.Stop();
            ts = stopWatch.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);


            // add label to Y
            Y = Y.Concatenate(labels);
           
            // write result to target path
            string targetPath = @"C:\Users\v-jiehu\source\repos\T-SNE\T-SNE\data";
            string fileName = "result.txt";

            string destFile = System.IO.Path.Combine(targetPath, fileName);
            if (System.IO.File.Exists(destFile))
            {
                // Use a try block to catch IOExceptions, to
                // handle the case of the file already being
                // opened by another process.
                try
                {
                    System.IO.File.Delete(destFile);
                }
                catch (System.IO.IOException e)
                {
                    Console.WriteLine(e.Message);
                }
            }

            if (!System.IO.File.Exists(targetPath))
            {
                System.IO.Directory.CreateDirectory(targetPath);
            }

            
            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(destFile))
            {

                for (var row=0;row< Y.GetLength(0);row++)
                {

                    var string_line = string.Join("\t", Y.GetRow(row));
                    file.WriteLine(string_line);
                }

            }


        }
    }
}
