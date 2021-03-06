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

/*! \file polynomial.h
 *  \brief polynomial relaxation.
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
class polynomial : public cusp::linear_operator<ValueType, MemorySpace>
{
public:

    // note: default_coefficients lives on the host
    cusp::array1d<ValueType, cusp::host_memory> default_coefficients;
    cusp::array1d<ValueType, MemorySpace> residual;
    cusp::array1d<ValueType, MemorySpace> h;
    cusp::array1d<ValueType, MemorySpace> y;

    polynomial();

    template <typename MatrixType, typename VectorType>
    polynomial(const MatrixType& A, const VectorType& coefficients);

    template <typename MemorySpace2>
    polynomial(const polynomial<ValueType,MemorySpace2>& A);

    template <typename MatrixType>
    polynomial(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level);

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);
   
    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x) const;

    template <typename MatrixType, typename VectorType1, typename VectorType2, typename VectorType3>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, VectorType3& coeffients);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/polynomial.inl>

