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

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

//   Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) * T
//   where K = diag(S)^-1 * S and rho(K) is an approximation to the
//   spectral radius of K.
template <typename MatrixType, typename ValueType>
void smooth_prolongator(const MatrixType& S,
                        const MatrixType& T,
                        MatrixType& P,
                        const ValueType omega,
                        const ValueType rho_Dinv_S);

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/smooth.inl>
