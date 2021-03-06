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

#include <cusp/convert.h>
#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////
        
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    hyb_matrix<IndexType,ValueType,MemorySpace>&
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

} // end namespace cusp

