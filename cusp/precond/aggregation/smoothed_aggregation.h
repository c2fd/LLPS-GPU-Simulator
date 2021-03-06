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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/spectral_radius.h>

#include <cusp/linear_operator.h>
#include <cusp/multilevel.h>

#include <cusp/precond/aggregation/smoothed_aggregation_options.h>

#include <cusp/relaxation/jacobi.h>

#include <vector> // TODO replace with host_vector

namespace cusp
{
namespace precond
{
namespace aggregation
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template<typename MatrixType>
struct sa_level
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    MatrixType A_; 					                              // matrix
    cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
    cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates

    ValueType rho_DinvA;

    sa_level() : rho_DinvA(0) {}

    template<typename SA_Level_Type>
    sa_level(const SA_Level_Type& sa_level)
      : A_(sa_level.A_), aggregates(sa_level.aggregates), B(sa_level.B), rho_DinvA(sa_level.rho_DinvA) {}
};


/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 */
template <typename IndexType, typename ValueType, typename MemorySpace,
	  typename SmootherType = cusp::relaxation::jacobi<ValueType,MemorySpace>,
	  typename SolverType = cusp::detail::lu_solver<ValueType,cusp::host_memory> >
class smoothed_aggregation :
  public cusp::multilevel< typename amg_container<IndexType,ValueType,MemorySpace>::solve_type, SmootherType, SolverType>
{

    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type SetupMatrixType;
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::solve_type SolveMatrixType;
    typedef typename cusp::multilevel<SolveMatrixType,SmootherType,SolverType> Parent;

public:

    const smoothed_aggregation_options<IndexType,ValueType,MemorySpace> & sa_options;
    const smoothed_aggregation_options<IndexType,ValueType,MemorySpace> default_sa_options;
    std::vector< sa_level<SetupMatrixType> > sa_levels;

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A);

    template <typename MatrixType, typename Options>
    smoothed_aggregation(const MatrixType& A,
                         const Options& sa_options);

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A, const cusp::array1d<ValueType,MemorySpace>& B);

    template <typename MatrixType, typename Options>
    smoothed_aggregation(const MatrixType& A, const cusp::array1d<ValueType,MemorySpace>& B,
                         const Options& sa_options);

    template <typename MemorySpace2,typename SmootherType2,typename SolverType2>
    smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2>& M);

protected:

    template <typename MatrixType, typename ArrayType>
    void sa_initialize(const MatrixType& A, const ArrayType& B);

    void extend_hierarchy(void);
};
/*! \}
 */

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/smoothed_aggregation.inl>

