using System.ComponentModel.DataAnnotations;

namespace taqa.polaris.sit.Models
{
    public class ReportInputModel
    {
        [Required]
        public string ToolId { get; set; }

        [Required]
        public string CustomerName { get; set; }

        [Required]
        public string WellName { get; set; }

        [Required]
        public DateTime StartDate { get; set; }

        [Required]
        public DateTime EndDate { get; set; }

        [Required]
        public string Namespace { get; set; }
    }
}
