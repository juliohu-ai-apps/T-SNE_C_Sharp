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


            String input_X = File.ReadAllText(@"C:\Users\v-jiehu\source\repos\T-SNE\T-SNE\data\mnist2500_X.txt");
            String input_labels = File.ReadAllText(@"C:\Users\v-jiehu\source\repos\T-SNE\T-SNE\data\mnist2500_labels.txt");
            var lines_X = input_X.Trim().Split('\n');
            var line1_X = lines_X[0].Split(new string[] { "   " }, StringSplitOptions.None);

            var m = Matrix.Create(lines_X.GetLength(0), line1_X.GetLength(0), 0.0);
            var r = 0;
            var c = 0;
            
            foreach(var line in lines_X)
            {
                
                foreach (var w in line.Trim().Split(new string[] { "   " }, StringSplitOptions.None))
                {
                    try
                    {
                        m[r,c] = Convert.ToDouble(w);
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

            var lines_Y = input_labels.Trim().Split('\n');
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
            Console.WriteLine("Reading files takes: " + ts.Seconds.ToString()+" seconds");
            stopWatch.Restart();
            var Y = new TSNE(m, 2, 2, 30.0)._TSNE();
            //var Y = X2P(data);
            stopWatch.Stop();
            ts = stopWatch.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);




        }
    }
}
