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

#include <cusp/exception.h>

#include <cusp/array1d.h>
#include <cusp/graph/breadth_first_search.h>

#include <thrust/scatter.h>
#include <thrust/iterator/constant_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace device
{

template<typename MatrixType, typename ArrayType>
size_t connected_components(const MatrixType& G, ArrayType& components)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType::index_type VertexId;

    const VertexId UNSET = -1;
    size_t num_components = 0;
    VertexId num_rows = G.num_rows;

    thrust::fill(components.begin(), components.end(), UNSET);
    ArrayType levels(G.num_rows, UNSET);
    VertexId src = rand() % G.num_rows;

    while(src < num_rows) {
        cusp::graph::breadth_first_search<false>(G, src, levels);
        thrust::transform_if( thrust::constant_iterator<VertexId>(num_components),
                              thrust::constant_iterator<VertexId>(num_components) + G.num_rows,
                              levels.begin(), components.begin(), thrust::identity<VertexId>(), _1 != UNSET );
    	thrust::fill(levels.begin(), levels.end(), UNSET);
        src = thrust::find(components.begin(), components.end(), UNSET) - components.begin();
        num_components++;
    }

    return num_components;
}

} // end namespace device
} // end namespace detail
} // end namespace graph
} // end namespace cusp
