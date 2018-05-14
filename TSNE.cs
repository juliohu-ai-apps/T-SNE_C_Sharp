/*                  T-SNE Class
 *  This is a C# simulate implementation of T-SNE with Python
 *  To use please add below "using" references
 *  
 *  Original Python code can be found at: https://lvdmaaten.github.io/tsne/ 
 * 
 */

using System;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.Math.Decompositions;
using System.Linq;

namespace T_SNE
{
    class TSNE
    {
        //
        //    Compute the perplexity and the P-row for a specific value of the
        //    precision of a Gaussian distribution.
        //

        private double[,] data;
        private int no_dims;
        private int initial_dims;
        private double perplexity;


        public TSNE(double[,] data,
                    int no_dims = 2,
                    int initial_dims = 50,
                    double perplexity = 30.0)
        {
            this.data = data;
            this.no_dims = no_dims;
            this.initial_dims = initial_dims;
            this.perplexity = perplexity;
        }


        public Tuple<double, double[]> Hbeta(double[] D, double beta = 1.0)
        {
            var P = Elementwise.Exp(Elementwise.Multiply(-beta, D.Copy()));
            var sumP = Matrix.Sum(P);
            var H = Math.Log(sumP) + Matrix.Sum(Elementwise.Multiply(D, P)) * beta / sumP;
            P = Elementwise.Divide(P, sumP);
            return Tuple.Create(H, P);
        }


        public double[,] X2P(double tol = 1e-5)
        {
            var X = data;
            Console.WriteLine("Computing pairwise distances...");
            var n = X.GetLength(0);
            var d = X.GetLength(1);


            var sum_X = Elementwise.Multiply(X, X).Sum(1);          //n x 1

            // how to use Elementwise Now? Since it's obsolete????
            var D = Elementwise.Add(Elementwise.Add(Elementwise.Multiply(X.DotWithTransposed(X), -2),
                                                    sum_X, VectorType.RowVector).Transpose(),
                                    sum_X, VectorType.RowVector);

            // P = np.zeros((n, n))
            var P = Matrix.Create(n, n, 0.0);
            // beta = np.ones((n, 1))
            var beta = Vector.Create(n, 1.0);
            // logU = np.log(perplexity)
            var logU = Math.Log(perplexity);

            for (var i = 0; i < n; i++)
            {
                if (i % 500 == 0)
                {
                    Console.WriteLine("Computing P-values for point {0} of {1}...", i, n);
                }

                var betamin = -1.0 / 0.0;
                var betamax = 1.0 / 0.0;

                // Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
                var Di = D.GetRow(i).RemoveAt(i);


                // (H, thisP) = Hbeta(Di, beta[i])
                var Tmp = Hbeta(Di, beta[i]);
                var H = Tmp.Item1;
                var thisP = Tmp.Item2;



                // Evaluate whether the perplexity is within tolerance
                var Hdiff = H - logU;
                var tries = 0;

                // while np.abs(Hdiff) > tol and tries < 50:
                while (Math.Abs(Hdiff) > tol && tries < 50)
                {
                    //if Hdiff > 0:
                    //    betamin = beta[i].copy()
                    //if betamax == np.inf or betamax == -np.inf:
                    //    beta[i] = beta[i] * 2.
                    //else:
                    //    beta[i] = (beta[i] + betamax) / 2.
                    if (Hdiff > 0)
                    {
                        betamin = beta[i];
                        if (betamax == 1.0 / 0.0 || betamax == -1.0 / 0.0)
                        {
                            beta[i] = beta[i] * 2.0;
                        }
                        else
                        {
                            beta[i] = (beta[i] + betamax) / 2.0;
                        }
                    }
                    //else:
                    //betamax = beta[i].copy()
                    //if betamin == np.inf or betamin == -np.inf:
                    //    beta[i] = beta[i] / 2.
                    //else:
                    //    beta[i] = (beta[i] + betamin) / 2.
                    else
                    {
                        betamax = beta[i];
                        if (betamin == 1.0 / 0.0 || betamin == -1.0 / 0.0)
                        {
                            beta[i] = beta[i] / 2.0;
                        }
                        else
                        {
                            beta[i] = (beta[i] + betamin) / 2.0;
                        }
                    }


                    //# Recompute the values
                    //(H, thisP) = Hbeta(Di, beta[i])
                    //Hdiff = H - logU
                    //tries += 1



                    Tmp = Hbeta(Di, beta[i]);
                    H = Tmp.Item1;
                    thisP = Tmp.Item2;
                    Hdiff = H - logU;
                    tries++;
                }


                //# Set the final row of P
                //P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP
                // P.Get()
                var k = 0;
                for (var j = 0; j < i; j++)
                {
                    P[i, j] = thisP[k];
                    k++;
                }

                for (var j = i + 1; j < n; j++)
                {
                    P[i, j] = thisP[k];
                    k++;
                }


            }
            //Return final P - matrix
            //print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
            Console.WriteLine("Mean value of sigma: {0}", beta.Pow(-1).Sqrt().Sum() / beta.Length);
            return P;

        }

        //
        //    """
        //        Runs PCA on the NxD array X in order to reduce its dimensionality to
        //        no_dims dimensions.
        //    """
        
        public double[,] PCA()
        {
            var X = data;
            Console.WriteLine("Preprocessing the data using PCA...");
            var n = X.GetLength(0);
            var d = X.GetLength(1);

            var mean_Vector = Elementwise.Divide(X.Sum(0), n * 1.0);


            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < d; j++)
                {
                    X[i, j] = X[i, j] - mean_Vector[j];
                }
            }

            var gevd = new EigenvalueDecomposition(X.Transpose().Dot(X));
            var M = gevd.Eigenvectors;
            // var l = gevd.RealEigenvalues;
            var Y = X.Dot(M.GetColumns(Vector.Range(no_dims)));

            return Y;
        }


        // np.add(lst1, lst2)
        public double[] NP_ADD(double[] lst1, double[] lst2)
        {
            var i = 0;
            // var j = 0;

            var min_length = lst1.Length > lst2.Length ? lst2.Length : lst1.Length;
            var result = Vector.Create(lst1.Length > lst2.Length ? lst1.Length : lst2.Length, 0.0);

            while (i < min_length)
            {
                result[i] = lst1[i] + lst2[i];
                i++;
            }

            while (i < lst1.Length)
            {
                result[i] = lst1[i];
                i++;
            }

            while (i < lst2.Length)
            {
                result[i] = lst2[i];
                i++;
            }
            return result;
        }

        // np.add(matrix, lst, dim=0)
        public double[,] NP_ADD(double[,] m, double[] lst, int n_dim = 0)
        {

            // check input
            if (n_dim != 0 && n_dim != 1)
            {
                Console.WriteLine("Wront n_dim, should be 0 or 1!");
                return new double[,] { { } };
            }

            if (n_dim == 0 && lst.Length != m.GetLength(1))
            {
                Console.WriteLine("Matrix column dimension not match with list length!");
                return new double[,] { { } };
            }

            if (n_dim == 1 && lst.Length != m.GetLength(0))
            {
                Console.WriteLine("Matrix row dimension not match with list length!");
                return new double[,] { { } };
            }

            if (n_dim == 0)
            {
                for (var i = 0; i < m.GetLength(0); i++)
                {
                    for (var j = 0; j < m.GetLength(1); j++)
                    {
                        m[i, j] = m[i, j] + lst[j];
                    }
                }
            }
            else  // n_dim == 1
            {
                for (var i = 0; i < m.GetLength(0); i++)
                {
                    for (var j = 0; j < m.GetLength(1); j++)
                    {
                        m[i, j] = m[i, j] + lst[i];
                    }
                }
            }

            return m;

        }



        //np.tile(lst, n_row, n_col)
        public double[,] NP_TILE(double[] lst, int repeats_row, int repeats_col)
        {


            var matrix = Matrix.Create(repeats_row, repeats_col * lst.Length, 0.0);

            for (var i = 0; i < repeats_row; i++)
            {
                for (var j = 0; j < lst.Length; j++)
                {
                    for (var r = 0; r < repeats_col; r++)
                    {
                        matrix[i, j + lst.Length * r] = lst[j];
                    }
                }
            }

            return matrix;
        }


        // np.maximum

        public double[,] NP_MAXIMUM(double[,] m, double num)
        {


            for (var i = 0; i < m.GetLength(0); i++)
            {
                for (var j = 0; j < m.GetLength(1); j++)
                {
                    if (m[i, j] < num)
                    {
                        m[i, j] = num;
                    }
                }
            }

            return m;
        }

        public double[,] _TSNE()
        {
            //# Initialize variables
            //    X = pca(X, initial_dims).real
            var X = PCA();
            var n = X.GetLength(0);
            var d = X.GetLength(1);
            var max_iter = 1000;

            var initial_momentum = 0.5;
            var final_momentum = 0.8;
            var eta = 500;
            var min_gain = 0.01;

            // Create a Normal with mean 2 and sigma 5
            var normal = new NormalDistribution(0, 1);

            // Generate samples from it
            double[] samples = normal.Generate(n * no_dims);

            var Y = samples.Reshape(n, no_dims);

            var dY = Matrix.Create(n, no_dims, 0.0);
            var iY = Matrix.Create(n, no_dims, 0.0);
            var gains = Matrix.Create(n, no_dims, 1.0);

            //# Compute P-values
            //P = x2p(X, 1e-5, perplexity)
            //P = P + np.transpose(P)
            //P = P / np.sum(P)
            //P = P * 4.									# early exaggeration
            //P = np.maximum(P, 1e-12)

            var P = X2P();
            P = P.Add(P.Transpose());
            P = P.Divide(P.Sum());
            P = P.Multiply(4.0);
            P = NP_MAXIMUM(P, 1e-12);



            for (var iter = 0; iter < max_iter; iter++)
            {
                //# Compute pairwise affinities
                //sum_Y = np.sum(np.square(Y), 1)
                //num = -2. * np.dot(Y, Y.T)
                //num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
                //num[range(n), range(n)] = 0.
                //Q = num / np.sum(num)
                //Q = np.maximum(Q, 1e-12)
                var sum_Y = Y.Multiply(Y).Sum(1);
                var num = Y.Dot(Y.Transpose()).Multiply(-2.0);




                num = NP_ADD(NP_ADD(num, sum_Y).Transpose(), sum_Y).Add(1.0).Pow(-1);

                for (var i = 0; i < num.GetLength(0); i++)
                {
                    num[i, i] = 0.0;
                }



                var Q = num.Divide(num.Sum());
                Q = NP_MAXIMUM(Q, 1e-12);
                var momentum = 0.0;

                //# Compute gradient
                //PQ = P - Q
                //for i in range(n):
                //    dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
                var PQ = P.Subtract(Q);


                for (var i = 0; i < n; i++)
                {
                    var YY = Matrix.Create(Y.GetLength(0), Y.GetLength(1), 0.0);
                    for (var r = 0; r < YY.GetLength(0); r++)
                    {
                        for (var c = 0; c < YY.GetLength(1); c++)
                        {
                            YY[r, c] = Y[i, c] - Y[r, c];
                        }
                    }

                    var T = NP_TILE(PQ.GetColumn(i).Multiply(num.GetColumn(i)), no_dims, 1).Transpose().Multiply(YY).Sum(0);



                    for (var c = 0; c < dY.GetLength(1); c++)
                    {
                        dY[i, c] = T[c];
                    }
                }





                if (iter < 20)
                {
                    momentum = initial_momentum;
                }

                else
                {
                    momentum = final_momentum;
                }

                //gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                //(gains * 0.8) * ((dY > 0.) == (iY > 0.))
                // gains[gains < min_gain] = min_gain

                var RowNum = gains.GetLength(0);
                var ColNum = gains.GetLength(1);
                for (var r = 0; r < RowNum; r++)
                {
                    for (var c = 0; c < ColNum; c++)
                    {
                        if ((dY[r, c] > 0.0 && iY[r, c] <= 0) || (dY[r, c] <= 0.0 && iY[r, c] > 0))
                        {
                            gains[r, c] = gains[r, c] + 0.2;
                        }
                        else
                        {
                            gains[r, c] = gains[r, c] * 0.8;
                        }

                        if (gains[r, c] < min_gain)
                        {
                            gains[r, c] = min_gain;
                        }
                        // iY = momentum * iY - eta * (gains * dY)
                        iY[r, c] = momentum * iY[r, c] - eta * (gains[r, c] * dY[r, c]);
                        Y[r, c] = Y[r, c] + iY[r, c];
                    }

                }
                // Y = Y - np.tile(np.mean(Y, 0), (n, 1))

                Y = Y.Subtract(NP_TILE(Y.Sum(0).Divide(Y.GetLength(0)), n, 1));



                //# Compute current value of cost function
                //if (iter + 1) % 10 == 0:
                //    C = np.sum(P * np.log(P / Q))
                //    print("Iteration %d: error is %f" % (iter + 1, C))
                if ((iter + 1) % 10 == 0)
                {

                    var C = P.Multiply(P.Divide(Q).Log()).Sum();
                    Console.WriteLine("Iteration {0}: error is {1}", (iter + 1), C);
                }


                //# Stop lying about P-values
                //if iter == 100:
                //    P = P / 4.
                if (iter == 100)
                {
                    P = P.Divide(4.0);
                }

            }
            return Y;
        }
    }
}
