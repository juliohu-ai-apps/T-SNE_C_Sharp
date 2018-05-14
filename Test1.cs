using System;
using T_SNE;

namespace MainTest
{
    class Program
    {

        static void Main(string[] args)
        {

            var data = new double [,] { { 1, 2, 3, 4, 5 },  
                                        { 2, 3, 4, 5, 6 },
                                        {7, 4, 3, 5, 7 },
                                        {6, 6, 7, 2, 4 },
                                        {5, 2, 3, 5, 7 } };


            var Y = new TSNE(data, 2, 2, 30.0)._TSNE();
            //var Y = X2P(data);



            for (var i = 0; i < Y.GetLength(0); i++)
            {
                for (var j = 0; j < Y.GetLength(1); j++)
                {
                    Console.Write(Y[i, j].ToString() + " ");
                }
                Console.Write("\n");
            }

        }
    }
}
