Ext.define('ClassificationApp.view.classification.ClassifyModel', {
    extend: 'Ext.app.ViewModel',
    alias: 'viewmodel.classify',
    data: {
        tooltip_refresh: 'Obnovit data',
        tooltip_new_planned_classification: 'Vytvořit novou plánovanou klasifikaci',
        tooltip_delete: 'Vymazání plánované klasifikace',
        tooltip_edit: 'Změna plánované klasifikace', 
        table_data: 'Dáta',
        talbe_model: 'Model',
        table_note: 'Poznámka',
        table_date: 'Dátum spuštění',
        table_email: 'E-mail',
        table_status: 'Status',
        table_error: 'Chyba',
        table_export: 'Export'
    }

});
