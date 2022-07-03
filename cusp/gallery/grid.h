 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file poisson.h
 *  \brief Poisson matrix generators
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/gallery/stencil.h>

namespace cusp
{
namespace gallery
{
/*! \addtogroup gallery Matrix Gallery
 *  \addtogroup grid Grid
 *  \ingroup gallery
 *  \{
 */

/*! \p grid2d: Create a matrix representing a 2d \p m by \p n grid.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns 
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/grid.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 * 
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     
 *     // create a matrix for a 4x4 grid
 *     cusp::gallery::grid2d(A, 4, 4);
 * 
 *     // print matrix
 *     cusp::print(A);
 * 
 *     return 0;
 * }
 * \endcode
 *
 */
template <typename MatrixType>
void grid2d(      MatrixType& matrix, size_t m, size_t n)
{
    CUSP_PROFILE_SCOPED();

    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType; 
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    stencil.push_back(StencilPoint(StencilIndex(  0, -1), 1));
    stencil.push_back(StencilPoint(StencilIndex( -1,  0), 1));
    stencil.push_back(StencilPoint(StencilIndex(  1,  0), 1));
    stencil.push_back(StencilPoint(StencilIndex(  0,  1), 1));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}
template <typename MatrixType>
void grid3d(      MatrixType& matrix, size_t m, size_t n, size_t l)
{
    CUSP_PROFILE_SCOPED();

    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType; 
    typedef thrust::tuple<IndexType,IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> 	    StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    for( IndexType k = -1; k <= 1; k++ )
    	for( IndexType j = -1; j <= 1; j++ )
    	   for( IndexType i = -1; i <= 1; i++ )
			   if(i==0 && j==0 && k== 0) continue;
    		   else stencil.push_back(StencilPoint(StencilIndex( i, j, k), 1));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n,l));
}
/*! \}
 */

} // end namespace gallery
} // end namespace cusp
