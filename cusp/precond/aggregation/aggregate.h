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

template <typename Matrix, typename Array>
void standard_aggregation(const Matrix& C, Array& aggregates);

template <typename Matrix, typename Array>
void standard_aggregation(const Matrix& C, Array& aggregates, Array& roots);

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/aggregate.inl>
