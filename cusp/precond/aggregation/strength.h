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

/*  Compute a strength of connection matrix using the standard symmetric measure.
 *  An off-diagonal connection A[i,j] is strong iff::
 *
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 *  With the default threshold (theta = 0.0) all connections are strong.
 *
 *  Note: explicit diagonal entries are always considered strong.
 */
template <typename Matrix1, typename Matrix2>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta = 0.0);

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/strength.inl>

