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

#include <cusp/graph/detail/dispatch/hilbert_curve.h>

namespace cusp
{
namespace graph
{

template <class Array2d, class Array1d>
void hilbert_curve(const Array2d& coord, const size_t num_parts, Array1d& parts)
{
    CUSP_PROFILE_SCOPED();

    return cusp::graph::detail::dispatch::hilbert_curve(coord, num_parts, parts,
					 typename Array2d::memory_space());
}

} // end namespace graph
} // end namespace cusp

