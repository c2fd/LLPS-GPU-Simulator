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

/*! \file jacobi.h
 *  \brief Jacobi relaxation.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/linear_operator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
// forward definitions
template<typename MatrixType> struct sa_level;
} // end namespace aggregation
} // end namespace precond

namespace relaxation
{

template <typename ValueType, typename MemorySpace>
class jacobi : public cusp::linear_operator<ValueType, MemorySpace>
{
public:
    ValueType default_omega;
    cusp::array1d<ValueType,MemorySpace> diagonal;
    cusp::array1d<ValueType,MemorySpace> temp;

    jacobi();

    template <typename MatrixType>
    jacobi(const MatrixType& A, ValueType omega=1.0);

    template <typename MemorySpace2>
    jacobi(const jacobi<ValueType,MemorySpace2>& A);

    template <typename MatrixType>
    jacobi(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level, ValueType weight=4.0/3.0);
    
    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);
   
    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);
        
    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, ValueType omega);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/jacobi.inl>

